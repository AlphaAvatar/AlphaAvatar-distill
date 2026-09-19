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

    assert RG.plan_hash(ROOT) == sha256_json(RG.plan_payload(ROOT))


def test_the_plan_hash_does_not_move_when_the_tree_does():
    """An authorization is issued against the plan hash and checked against it
    at launch, with at least one commit in between — the commit that carries
    the authorization itself.

    The first version hashed the whole source binding, which records the LIVE
    `head` beside the source commit. So the plan hash moved with every commit
    and the binding could never hold from issuance to launch. It failed at the
    launcher, after the sweep, the authorization and the bundle had been built
    on it.
    """
    payload = RG.plan_payload(ROOT)
    flat = json.dumps(payload)
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    assert head, "no HEAD to compare against"
    assert head not in flat, (
        "the plan payload contains the live HEAD, so the plan hash moves with "
        "every commit and no authorization can survive to launch")
    #: The attempt-3 commit SHOULD be in it — that one is the plan.
    assert RS.ATTEMPT3_SESSION_COMMIT in flat
    assert RS.SELECTION_SHA256 in flat


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


def test_the_session_fits_its_money_while_there_is_work_left():
    """The planner refuses a plan it cannot fund. It refused once already.

    Conditional on whether work REMAINS, because the question changes when the
    campaign finishes: once every selected leaf is reconstructed and durable,
    demanding that the remaining money still fund a full five-path run would
    require the campaign to fund a sixth run nobody needs. The condition is
    derived from the durable store, not from a flag somebody sets.
    """
    import autoinit_c2_replay_launch as L

    money = RG.remaining_usd(ROOT)
    outstanding = [leaf for leaf in RS.build_replay_leaves(ROOT, device="cpu")
                   if not (Path(L.DURABLE_STORE) / leaf.state_id
                           / "durable_ack.json").is_file()]
    if not outstanding:
        #: Campaign complete. What is left is what is left.
        assert money["remaining_all_in_usd"] >= 0, money
        return

    assert money["this_attempt_gpu_usd"] > 0, (
        f"{len(outstanding)} leaves are still outstanding and the campaign has "
        "nothing left to authorize; that is a maintainer decision, not a test "
        "failure")
    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40,
         "--bundle", "b.bundle", "--run-id", "preflight"])
    plan = L.budget(args).plan(price_per_hour=1.09,
                               authorized_usd=money["this_attempt_gpu_usd"])
    assert plan.hard_terminate_minutes / 60.0 * 1.09 <= money["this_attempt_gpu_usd"]
    assert plan.soft_stop_minutes < plan.hard_terminate_minutes


def test_every_selected_leaf_is_durable_and_exact():
    """The deliverable, checked against the FROZEN SELECTION rather than
    against any run's claims about itself."""
    import autoinit_c2_replay_launch as L

    store = Path(L.DURABLE_STORE)
    selection = {e["state_id"]: e for e in RS.load_selection(ROOT)["selected"]}
    missing = [sid for sid in selection
               if not (store / sid / "durable_ack.json").is_file()]
    if missing:
        #: Not a failure before the work is done — this file is also the
        #: PREFLIGHT for a run that has not happened yet.
        return
    for sid, entry in selection.items():
        sidecar = json.loads((store / sid / "replay_leaf.json").read_text())
        assert sidecar["identity"]["artifact_digest"] == entry["artifact_digest"]
        assert sidecar["identity"]["single_shard_sha256"] == \
            entry["single_shard_sha256"]
        assert sidecar["identity"]["num_parameters"] == entry["num_parameters"]
        assert sidecar["identity"]["tokenizer_sha256"] is None
        assert (store / sid / "model.safetensors").stat().st_size == 1_192_135_096


def test_the_attempt_is_authorized_for_the_campaign_remainder_not_its_ceiling():
    """A cumulative ceiling read as a per-attempt allowance is not a ceiling.

    Nine attempts at $4.77 each would be $43 against a $4.77 campaign. The
    remainder is DERIVED from the recorded closeouts, so it cannot drift from
    what was actually spent.
    """
    money = RG.remaining_usd(ROOT)
    assert money["campaign_all_in_usd"] == RG.CAMPAIGN_ALL_IN_USD
    assert money["campaign_spent_usd"] > 0, (
        "no spend is visible; either the closeouts moved or this campaign has "
        "not run, and a fresh campaign should not be reading this test")
    assert (money["remaining_all_in_usd"]
            == round(RG.CAMPAIGN_ALL_IN_USD - money["campaign_spent_usd"], 4))
    assert money["this_attempt_gpu_usd"] < RG.CAMPAIGN_GPU_USD


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


def test_the_authorization_document_round_trips_through_its_own_loader(tmp_path):
    """Assemble one, then load it. The two halves were written independently
    and did not agree.

    `as_dict` serialises `plan_hash` as `phase_a_session_plan_hash` and
    `science_plan_hash` as `phase_a_science_plan_hash`, and the constructor
    requires eleven fields. A document assembled from what this session felt it
    needed parsed as nothing the loader could read — and the failure surfaced in
    the dry run, AFTER the sweep, the authorization and the bundle had all been
    built on it, costing a whole governance chain.

    Built here rather than read off the tree, because the sweep runs BEFORE the
    authorization exists: a preflight that asserted the artifact could never
    pass on the tree it is meant to certify.
    """
    import json as _json

    from issue_c2_replay_authorization import build_authorization_record

    grant = _json.loads(
        (ROOT / "logs/stages/stage-1/phase_c2_replay/runs/attempt1"
                "/governance/grant.json").read_text())
    record = build_authorization_record(
        grant=grant, approved=grant["approved_money"],
        commit="d" * 40, dirty=False, rate=1.09,
        binding=RS.source_binding(ROOT), live=RG.current_executable(ROOT),
        plan_hash=RG.plan_hash(ROOT))

    path = tmp_path / "authorization.json"
    path.write_text(_json.dumps(record, indent=1) + "\n")
    a = RG.ReplayAuthorization.load(path)

    assert a.authorization_id == RG.PLAN_ID
    assert tuple(a.authorized_stages) == RG.AUTHORIZED_STAGES
    #: The REMAINDER, not the campaign ceiling.
    assert a.hard_cap_usd == RG.remaining_usd(ROOT)["this_attempt_gpu_usd"]
    assert a.plan_hash == RG.plan_hash(ROOT)
    assert a.authorizes_c2_replay is True
    assert a.authorizes_c2_full_search is False
    assert a.harness_source_digest and a.harness_source_files
    #: And the money the type does not carry.
    assert record["money"]["all_in_usd"] == (
        RG.remaining_usd(ROOT)["remaining_all_in_usd"])
    assert record["money"]["campaign_all_in_usd"] == RG.CAMPAIGN_ALL_IN_USD


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


def test_the_launcher_records_its_run_with_what_record_run_requires():
    """`record_run` is keyword-only and requires plan, implementation, status
    and roles. Called with the spec alone it raises — inside the `finally` that
    exists to record a failed run, so the failure path itself failed and the
    attempt id was burned."""
    import inspect

    import autoinit_c2_replay_launch as L
    from experiments.run_layout import record_run

    required = {name for name, p in
                inspect.signature(record_run).parameters.items()
                if p.kind is p.KEYWORD_ONLY
                and p.default is inspect.Parameter.empty}
    source = inspect.getsource(L.main)
    for name in sorted(required):
        assert f"{name}=" in source, (
            f"record_run requires {sorted(required)}; main() passes no {name}=")


def test_every_transport_name_the_launcher_uses_exists_and_is_callable():
    """The launcher reaches into the bundle module by attribute.

    `bundle_staged_gate` called `RT.roundtrip` and the module had no such name,
    so the launch aborted at the seventh gate after the whole chain had been
    issued. Checked by reading the launcher's source for `RT.<name>` rather
    than by listing names somebody remembered.
    """
    import inspect
    import re as _re

    import autoinit_c2_replay_launch as L
    from experiments.phase_c2 import replay_bundle as RT

    used = set(_re.findall(r"\bRT\.([A-Za-z_][A-Za-z0-9_]*)",
                           inspect.getsource(L)))
    assert used, "the probe found no RT.<name> uses; it is not checking anything"
    for name in sorted(used):
        attr = getattr(RT, name, None)
        assert attr is not None, f"replay_bundle has no {name!r}"
    #: And the one that actually transports must accept what the gate passes.
    params = inspect.signature(RT.roundtrip).parameters
    for needed in ("session_commit", "local_bundle_sha256",
                   "authorization_bytes", "authorization_path",
                   "expected_harness_digest", "harness_files", "workdir"):
        assert needed in params, f"roundtrip takes no {needed}"


def test_the_bound_image_is_an_image_reference():
    """It was `POD_IMAGE`, which is the deployment-commands MAPPING — workspace
    roots, remote python, min CUDA — not an image reference. That put a dict on
    the `runpodctl pod create` command line, and it surfaced inside `create()`,
    which is one line away from a billing resource."""
    import autoinit_c2_replay_launch as L

    assert isinstance(L.BOUND_IMAGE, str), type(L.BOUND_IMAGE)
    assert "/" in L.BOUND_IMAGE and ":" in L.BOUND_IMAGE, L.BOUND_IMAGE
    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40,
         "--bundle", "b.bundle", "--run-id", "preflight"])
    assert args.image == L.BOUND_IMAGE


def test_the_dry_run_flag_actually_stops_before_provider_creation():
    """The flag is declared by this launcher and honoured by the shared runner.

    It was honoured by nothing: `SessionRunner.run` went from the prechecks
    straight to `create()`, so a dry run whose gates all passed created a real
    pod.
    """
    import inspect

    from aadistill.infrastructure.session_runner import SessionRunner

    import autoinit_c2_replay_launch as L

    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40,
         "--bundle", "b.bundle", "--run-id", "preflight", "--dry-run"])
    assert args.dry_run is True
    source = inspect.getsource(SessionRunner.run)
    assert "dry_run" in source, (
        "the runner does not consult dry_run; this launcher's --dry-run would "
        "create a provider resource")
    assert source.index("run_prechecks") < source.index('"dry_run"')


def test_the_pod_paths_agree_with_the_setup_script():
    """The launcher's pod paths are the SCRIPT's, read from it, not invented.

    This file first declared `WS=/workspace/aad` and `REPO=/workspace/aad/repo`
    and put the status file under a `scratch/` subdirectory nothing creates.
    The script's very first `mark()` writes to that file, so setup died one line
    after ENV_READY — 1.2 minutes and $0.02 into a billing pod, for a path that
    could have been read here.
    """
    import re as _re

    import autoinit_c2_replay_launch as L

    text = (ROOT / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    ws = _re.search(r"^WS=(\S+)", text, _re.M)
    repo = _re.search(r"^REPO=(\S+)", text, _re.M)
    assert ws and repo, "the setup script no longer declares WS and REPO"
    assert L.WS == ws.group(1), (L.WS, ws.group(1))
    assert L.REPO == repo.group(1).replace("$WS", L.WS), (L.REPO, repo.group(1))

    #: The status and run log live directly in WS. `mark()` appends to the
    #: status file before anything has created a subdirectory, so a path with
    #: one in it cannot work.
    for path in (L.STATUS, L.RUN_LOG):
        assert path.startswith(L.WS + "/"), path
        assert "/" not in path[len(L.WS) + 1:], (
            f"{path} puts the file in a subdirectory of {L.WS}; nothing creates "
            "one before the script's first mark()")

    #: And everything the driver writes lives under the checkout.
    for path in (L.WORKDIR, L.LEAF_DIR):
        assert path.startswith(L.REPO + "/"), path


def test_the_driver_writes_its_evidence_where_the_collector_looks():
    """One required artifact, two places that must name the same path.

    The artifact spec patterns the evidence at `audit/<dirname>/...` under the
    pod's artifacts root; the driver first wrote it into its workdir. It is the
    session's ONE required artifact, so a torn-down run would have come home
    with no record of itself — and on a failure path the evidence is the only
    thing worth having.

    Both ends are read here, not restated: the spec's pattern and the launcher's
    driver command.
    """
    from collect_artifacts import load_specs

    import autoinit_c2_replay_launch as L

    specs = load_specs(str(ROOT / "configs/autoinit/c2_replay_artifacts.json"))
    required = [x for x in specs if x.required]
    assert len(required) == 1, [x.pattern for x in required]
    pattern = required[0].pattern
    assert pattern.startswith(f"audit/{L.AUDIT_DIRNAME}/"), pattern

    #: `AUDIT_DIR` is the absolute form of that same pattern on the pod.
    assert L.AUDIT_DIR == f"{L.REPO}/artifacts/audit/{L.AUDIT_DIRNAME}"
    assert L.AUDIT_DIR.endswith("/" + pattern.rsplit("/", 1)[0])

    #: And the launcher must actually hand it to the driver.
    ctx = type("C", (), {"image_digest": "", "price": 1.09, "spent_usd": 0.0,
                         "args": L.build_parser().parse_args(
                             ["--scr", "/tmp/x", "--session-commit", "d" * 40,
                              "--bundle", "b.bundle", "--run-id", "preflight"]),
                         "auth": type("A", (), {"hard_cap_usd": 4.69})()})()
    plan = type("P", (), {"soft_stop_usd": 4.41})()
    command = L.driver_command(ctx, plan)
    assert f"--audit-dir {L.AUDIT_DIR}" in command, command


def test_every_marker_this_session_declares_has_what_the_script_demands():
    """DERIVED from the setup script, not a list of the requirements we know.

    The script guards blocks with `step_declared <MARKER>` and inside them uses
    `${SESSION_X:?why}` to refuse a session that declared the marker without
    naming what the block needs. Attempt 7 declared ASSETS_READY and named no
    frozen-asset expectation: setup died at that line, 2.7 minutes and $0.05
    into a billing pod, for a requirement written in the script it was running.

    This parses the guards out of the script, so a requirement added tomorrow is
    checked without anyone remembering to add a test for it.
    """
    import re as _re

    import autoinit_c2_replay_launch as L

    text = (ROOT / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    spec = L.spec(L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40,
         "--bundle", "b.bundle", "--run-id", "preflight"]))
    declared = set(spec.setup.setup_markers)
    provided = dict(spec.setup.env)

    #: Split the script into `step_declared <MARKER>` blocks.
    blocks = _re.split(r"^if step_declared (\w+); then$", text, flags=_re.M)
    found_any = False
    for i in range(1, len(blocks), 2):
        marker, body = blocks[i], blocks[i + 1]
        if marker not in declared:
            continue
        for var in _re.findall(r'[:"]?\$\{(SESSION_\w+):\?', body):
            found_any = True
            assert var in provided or var in spec.setup.required_env, (
                f"the script refuses a session that declares {marker} without "
                f"{var}; this session declares {marker} and provides neither "
                f"env nor required_env for it")
    assert found_any, (
        "the probe matched no requirements at all; either the script changed "
        "shape or this test is checking nothing")

    #: And the document it names must exist and be the one for THIS session.
    expect = ROOT / provided["SESSION_FROZEN_EXPECT"]
    assert expect.is_file(), expect


def test_the_frozen_asset_expectation_names_what_the_five_paths_read():
    """Not Search-1's document, and not more than this session stages."""
    import json as _json

    import autoinit_c2_replay_launch as L

    doc = _json.loads((ROOT / L.FROZEN_EXPECT).read_text())
    roots = {e["root"] for e in doc["assets"].values()}
    assert roots == {a.repo_path for a in L.LOCAL_ASSETS}, (
        roots, [a.repo_path for a in L.LOCAL_ASSETS])
    assert "scoring_contract" not in doc, (
        "the replay consumes no scoring contract; naming one would make the "
        "verifier check a thing this session does not have")

    #: The content hashes must be the mixtures the SELECTION was searched
    #: against — the same values the frozen selection records.
    declared = {p["content_sha256"] for p in RS.load_selection(ROOT)["profiles"]}
    assert {e["content_sha256"] for e in doc["assets"].values()} == declared


def test_the_frozen_asset_verifier_passes_against_this_expectation():
    """Run the real verifier. A document that merely parses is not evidence."""
    import subprocess
    import tempfile

    import autoinit_c2_replay_launch as L

    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [sys.executable, "scripts/autoinit/verify_frozen_assets.py",
             "--expect", L.FROZEN_EXPECT,
             "--out", str(Path(tmp) / "check.json")],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
            env={**__import__("os").environ, "PYTHONPATH": "src:scripts"})
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert '"passed": true' in out.stdout


def test_the_drivers_markers_reach_the_file_the_launcher_tails(tmp_path):
    """The runner decides a session's terminal from the STATUS FILE.

    Attempt 9 reconstructed all five leaves exactly and was recorded
    INCOMPLETE: the driver printed C2_REPLAY_ALL_DONE to stdout, the runner
    tails the status file and never saw it, and the session was classified by
    the exit code instead. The setup script appends to the same file, which is
    why SETUP_DONE was visible and nothing after it was.
    """
    import autoinit_c2_replay_driver as D
    import autoinit_c2_replay_launch as L

    status = tmp_path / "session.status"
    saved = D.STATUS_PATH
    try:
        D.STATUS_PATH = status
        D.mark(D.SUCCESS_MARKER)
        D.mark(D.LEAF_MARKER, "abc123")
    finally:
        D.STATUS_PATH = saved

    written = status.read_text()
    assert f"MARKER:{D.SUCCESS_MARKER}" in written
    assert f"MARKER:{D.LEAF_MARKER}:abc123" in written

    #: And the launcher must hand the driver that path.
    ctx = type("C", (), {"image_digest": "", "price": 1.09, "spent_usd": 0.0,
                         "args": L.build_parser().parse_args(
                             ["--scr", "/tmp/x", "--session-commit", "d" * 40,
                              "--bundle", "b.bundle", "--run-id", "preflight"]),
                         "auth": type("A", (), {"hard_cap_usd": 3.49})()})()
    command = L.driver_command(ctx, type("P", (), {"soft_stop_usd": 2.85})())
    assert f"--status-path {L.STATUS}" in command, command


def test_a_marker_write_failure_does_not_kill_the_driver(tmp_path):
    """A driver that died because it could not append to a status file would
    lose the work the file exists to report."""
    import autoinit_c2_replay_driver as D

    saved = D.STATUS_PATH
    try:
        D.STATUS_PATH = tmp_path / "no" / "such" / "dir" / "session.status"
        D.mark(D.FAILURE_MARKER)          # must not raise
    finally:
        D.STATUS_PATH = saved


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

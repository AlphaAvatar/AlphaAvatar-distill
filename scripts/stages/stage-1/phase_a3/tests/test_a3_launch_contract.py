"""What the A3 launcher promises before a pod exists, checked at `$0`.

The rehearsal next door drives the driver. This checks the LAUNCHER: the
session identity, the dispatch the pod will take, the evidence declaration
that decides what survives, the budget that bounds the spend, and each gate's
refusal. Everything here is free, and every item on it has cost this
programme money somewhere else.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "pod"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stages.phase_a3 import a3_session as A3S  # noqa: E402

SETUP_SCRIPT = REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"


def launcher():
    spec = importlib.util.spec_from_file_location(
        "a3lau_contract", REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["a3lau_contract"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def session():
    L = launcher()
    #: Through the launcher's own parser with the flags the ACQUISITION LOOP
    #: sends, not a convenient subset. The subset is what hid the missing
    #: `--scr`/`--session-commit`/`--bundle` declarations: every test here
    #: built a namespace the real command line could not have produced.
    args = L.build_parser().parse_args(
        ["--scr", "/tmp/a3-contract", "--run-id", "a3_attempt1",
         "--session-commit", "0" * 40, "--bundle", "aad_test.bundle",
         "--gpu", "NVIDIA L40S", "--max-price", "1.09"])
    args.disk_gb = 60
    return L, args, L.spec(args)


# --- the dispatch the pod will actually take ------------------------------


def test_the_setup_script_has_a_branch_for_this_session_kind(session):
    """A missing SESSION_KIND branch is not a type error -- it is $0.23.

    It falls through to `spend`, which loads a SpendAuthorization and exits 98
    AFTER setup has run on a billing pod.
    """
    _, _, spec = session
    kind = spec.setup.env["SESSION_KIND"]
    script = SETUP_SCRIPT.read_text()
    assert f'"$SESSION_KIND" = "{kind}"' in script, (
        f"the shared setup script has no branch for SESSION_KIND={kind!r}")
    branch = script.split(f'"$SESSION_KIND" = "{kind}"')[1].split("elif [")[0]
    #: And the branch must carry the governance boundary INTO the pod.
    assert "A3Authorization" in branch
    for refused in ("authorizes_c1_isolation is False",
                    "authorizes_c3_isolation is False",
                    "allows_beam_search is False",
                    "allows_arm_elimination is False",
                    "allows_control_retraining is False",
                    "allows_on_pod_decision is False"):
        assert refused in branch, refused
    assert "authorizes_a3 is True" in branch


def test_the_declared_setup_markers_match_what_the_script_can_emit(session):
    """Every declared step must be one the shared script can actually run.

    Matched on the marker NAME, not on an exact `mark NAME` line: the script
    legitimately emits `mark "TESTS_OK:${tt}s"` with the elapsed time
    appended, and a literal-line assertion would force the declaration to
    drift rather than catch a real gap.
    """
    _, _, spec = session
    script = SETUP_SCRIPT.read_text()
    import re
    emitted = set(re.findall(r'mark "?([A-Z_]+)', script))
    declared_steps = set(re.findall(r'step_declared ([A-Z_]+)', script))
    for marker in spec.setup.setup_markers:
        assert marker in emitted, (
            f"the session declares the marker {marker!r} and the shared setup "
            "script never emits it, so setup would wait for a step that "
            "cannot happen")
    #: And every OPTIONAL step the session declares must be gated on the
    #: declaration, or it would run for sessions that never asked for it.
    for marker in spec.setup.setup_markers:
        if marker in ("ENV_READY", "REPO_READY", "SETUP_DONE"):
            continue
        assert marker in declared_steps or marker in emitted, marker


def test_rope_ok_is_declared_and_its_input_is_staged(session):
    """C1 attempt 2 declared the three tokenizer sidecars and nothing else, so
    the ROPE_OK glob was empty and setup exited AFTER the teacher had been
    fetched and verified -- $0.1013."""
    _, _, spec = session
    assert "ROPE_OK" in spec.setup.setup_markers
    staged = {r.path for r in spec.setup.relay_inputs}
    assert any(p.endswith("/checkpoint/config.json") for p in staged), (
        "ROPE_OK globs artifacts/stages/stage-1/*/checkpoint/config.json and no "
        "relay input stages one")


def test_vllm_is_declared_because_a3_generates(session):
    """A3 evaluates three probes, so it serves. A session that declared no
    VLLM_READY would not build the inference environment."""
    _, _, spec = session
    assert "VLLM_READY" in spec.setup.setup_markers


# --- one owner for the status path and the job id -------------------------


def test_the_status_path_and_job_id_come_from_the_one_owner(session):
    """The C3 pair hardcoded two different status files, so every driver
    marker -- ALL_DONE included -- was invisible to the launcher, and the
    acquisition loop would have read a completed formal run as 'no
    measurement began' and launched a second paid attempt."""
    _, _, spec = session
    from stages.phase_a3 import autoinit_a3_driver as D

    assert spec.status_path == A3S.STATUS_PATH
    assert spec.driver_job_id == A3S.DRIVER_JOB_ID
    #: And the DRIVER writes to the same file the launcher polls.
    assert str(D.STATUS) == A3S.STATUS_PATH
    assert spec.run_log_path == A3S.RUN_LOG_PATH


def test_the_markers_the_launcher_watches_are_the_markers_the_driver_emits(
        session):
    _, _, spec = session
    src = (REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_driver.py").read_text()
    assert f'mark("{spec.markers.success}")' in src, (
        f"the launcher waits for {spec.markers.success!r} and the driver "
        "never emits it")
    for failure in spec.markers.failure:
        assert f'mark("{failure}")' in src, (
            f"the launcher classifies on {failure!r} and the driver never "
            "emits it")


def test_the_failure_note_asserts_nothing_about_reproducibility(session):
    """`SessionRunner` prints this for ANY failure marker. C3's text described
    a replay mismatch whatever had happened -- and on one attempt a stage
    raised before a single digest was computed while the launcher announced
    that the frozen path had not reproduced."""
    _, _, spec = session
    note = spec.markers.failure_note.lower()
    assert "asserts nothing" in note
    assert "reproduce" in note


# --- the evidence declaration that decides what survives ------------------


def test_the_artifact_specs_are_scaled_to_three_probes(session):
    _, _, spec = session
    success = json.loads((REPO / spec.artifacts.spec_success).read_text())
    n = A3S.A3_SESSION_CONTRACT.n_probes
    per_probe = [e for e in success["entries"]
                 if e.get("required") and "A_bsz3.*" in e["pattern"]]
    assert per_probe, "no per-probe entry is required at all"
    for e in per_probe:
        assert e["min_matches"] % n == 0, (
            f"{e['pattern']} requires {e['min_matches']}, not a multiple of "
            f"{n}; a spec scaled to another design classifies a complete run "
            "as incomplete")
    #: Seven sets x three probes for the generation files.
    gens = [e for e in per_probe if "generations.jsonl" in e["pattern"]]
    assert gens and gens[0]["min_matches"] == 21, (
        "requiring 3 would accept a probe that generated one set of seven")


def test_the_success_spec_requires_no_decision_artifact(session):
    """A3 computes none on the pod. Requiring one would classify a COMPLETE
    measurement as a failure -- which is how attempt75's nine scored probes
    came home under the failed spec."""
    _, _, spec = session
    success = json.loads((REPO / spec.artifacts.spec_success).read_text())
    assert not [e for e in success["entries"] if "decision" in e["pattern"]]
    assert A3S.AGGREGATION_IS_OFF_POD is True
    assert not any(s.letter == "I" for s in A3S.STAGES)


def test_the_failed_spec_requires_nothing(session):
    """An early failure may legitimately have produced none of it, and
    demanding a product on a failure path blocks the collection of what the
    run DOES have."""
    _, _, spec = session
    failed = json.loads((REPO / spec.artifacts.spec_failed).read_text())
    assert failed["entries"]
    assert not any(e.get("required") for e in failed["entries"])
    assert all(e.get("min_matches", 0) == 0 for e in failed["entries"])


def test_both_specs_are_inside_the_harness_the_grant_measures(session):
    """Without this, editing an evidence declaration would not move the
    harness digest and a grant would certify a collection policy it never
    saw."""
    from stages.phase_a3.a3_authorization import A3_HARNESS_FILES

    _, _, spec = session
    for path in (spec.artifacts.spec_success, spec.artifacts.spec_failed):
        assert path in A3_HARNESS_FILES, path


def test_every_report_name_is_a_file_the_driver_writes(session):
    _, _, spec = session
    src = (REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_driver.py").read_text()
    for name in spec.artifacts.report_names:
        assert name in src, (
            f"the artifact policy reports {name!r} and the driver never "
            "writes it")


# --- the budget bounds the spend exactly ----------------------------------


def test_the_plan_reproduces_the_derived_ceiling_rather_than_approaching_it(
        session):
    """A planner that applied its own contingency on top of the component
    table's overrun factor once produced a plan terminating ABOVE its grant."""
    from stages.phase_a3.a3_authorization import (
        a3_billed_rate_usd_per_hour, a3_hard_ceiling_usd,
    )

    _, _, spec = session
    rate = a3_billed_rate_usd_per_hour(REPO)
    ceiling = a3_hard_ceiling_usd(REPO)
    plan = spec.budget.plan(price_per_hour=rate, authorized_usd=ceiling)
    assert plan.hard_terminate_usd <= ceiling + 1e-4, (
        f"the plan terminates at ${plan.hard_terminate_usd:.4f} against an "
        f"authorized ${ceiling:.4f}")
    assert plan.hard_terminate_usd > ceiling - 0.01, (
        "the plan terminates well below its ceiling, so the components and "
        "the price have drifted apart")
    #: The soft stop leaves a teardown's worth, which P12.1 requires.
    assert plan.soft_stop_usd < plan.hard_terminate_usd
    assert plan.artifact_recovery_reserve_minutes > 0
    assert spec.budget.contingency_fraction == 0.0, (
        "the component table already carries the overrun allowance; a second "
        "multiplier prices the same risk twice")


def test_the_session_declares_the_shape_the_design_froze(session):
    _, _, spec = session
    ev = spec.evidence_fields
    assert ev["arms"] == 1 and ev["probes"] == 3
    assert ev["seeds"] == len(A3S.recovery_seeds())
    assert ev["runs_a_search"] is False
    assert ev["eliminates_arms"] is False
    assert ev["retrains_controls"] is False
    assert ev["computes_a_decision_on_pod"] is False
    assert ev["gated_protocol"] == A3S.REFERENCE_PROTOCOL
    assert ev["measured_protocol"] == A3S.TREATMENT_PROTOCOL
    assert ev["a3_design_sha256"] == A3S.design_hash()
    assert spec.plan_hash == A3S.A3_SESSION_CONTRACT.contract_hash


# --- each gate refuses for its own reason ---------------------------------


def _ctx(args, **over):
    #: The ceiling is DERIVED, never a literal. It was `8.2525`, and the
    #: measured container-disk repair moved it to `8.3047` -- so a test about
    #: the RATE check started failing on the ceiling check instead, with a
    #: message about something it was not asserting. A literal here is a
    #: second owner of a derived number.
    from stages.phase_a3.a3_authorization import a3_hard_ceiling_usd

    base = dict(args=args, auth=types.SimpleNamespace(
        hard_cap_usd=a3_hard_ceiling_usd(REPO), harness_source_digest="x" * 64,
        harness_source_files=()), evidence={}, say=lambda *_: None)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_the_harness_gate_refuses_a_digest_that_does_not_match(session):
    L, args, _ = session
    #: A3 is closed and A3_HARNESS_FILES keeps its freeze-time spellings; the
    #: 2026-10-08 migration moved its members, so the gate now refuses one
    #: step earlier — the declaration itself no longer resolves and the
    #: digest helper raises — which is the historical-declarations contract,
    #: and still fails closed.
    from aadistill.governance.authorization import AuthorizationError

    try:
        ok, why = L.a3_harness_gate(_ctx(args))
    except AuthorizationError as exc:
        assert "is missing" in str(exc)
    else:
        assert ok is False and "harness digest" in why
    #: And a grant naming no digest at all is refused, not defaulted.
    ctx = _ctx(args, auth=types.SimpleNamespace(
        hard_cap_usd=8.2525, harness_source_digest="", harness_source_files=()))
    ok, why = L.a3_harness_gate(ctx)
    assert ok is False and "names no harness digest" in why


#: `test_the_controls_evidence_gate_passes_on_the_real_evidence` MOVED to
#: tests/autoinit/test_a3_controls_evidence.py. This directory is A3's pod
#: test selection, and that test's premise -- attempt75's durable evidence
#: being mounted -- is a dev-box fact at an absolute path outside the
#: repository, so the simulator cannot hide it: it RAN in the launch-bound
#: sweep and SKIPPED on the pod, the skip sets differed by one nodeid, and
#: the shared setup refused with `50 passed, 5 skipped, 0 failed`. The gate
#: is unaffected and still runs on the pod; only the test moved.


def test_the_battery_gate_requires_the_controls_own_battery(session):
    L, args, _ = session
    ctx = _ctx(args)
    ok, why = L.battery_staged_gate(ctx)
    if not ok and "not staged" in why:
        pytest.skip("the confirmation battery is not staged here")
    assert ok, why
    ev = ctx.evidence["battery"]
    assert ev["content_sha256"] == ev["expected_content_sha256"]


def test_the_design_gate_reads_a_hash_bound_design(session):
    L, args, _ = session
    ctx = _ctx(args)
    ok, why = L.a3_design_gate(ctx)
    assert ok, why
    assert ctx.evidence["a3_design"]["probes"] == 3
    assert ctx.evidence["a3_design"]["aggregation_off_pod"] is True


def test_the_pricing_gate_refuses_a_ceiling_the_grant_does_not_carry(session):
    L, args, _ = session
    ctx = _ctx(args, auth=types.SimpleNamespace(
        hard_cap_usd=30.0, harness_source_digest="x" * 64,
        harness_source_files=()))
    ok, why = L.pricing_identity_gate(ctx)
    assert ok is False
    assert "DERIVED ceiling, never an envelope" in why


def test_the_pricing_gate_refuses_a_rate_above_what_was_priced(session):
    L, _, _ = session
    args = L.build_parser().parse_args(
        ["--scr", "/tmp/a3-contract", "--run-id", "a3_attempt1",
         "--session-commit", "0" * 40, "--bundle", "aad_test.bundle",
         "--max-price", "2.50"])
    args.disk_gb = 60
    ok, why = L.pricing_identity_gate(_ctx(args))
    assert ok is False and "exceeds the priced rate" in why


def test_the_gates_are_ordered_so_the_network_ones_are_last(session):
    """A cheap local refusal must still cost nothing."""
    _, _, spec = session
    names = [getattr(g, "__name__", type(g).__name__) for g in spec.precheck]
    assert names[-1] == "no_billing_resource_gate", names
    #: The same-failure rule sits immediately before the billing check,
    #: because both answer "may a provider resource exist at all" and a
    #: repetition is as wasteful as a second concurrent pod.
    assert names[-2] == "same_failure_gate", names
    assert names[-3] == "durable_capacity_gate", names
    assert "controls_evidence_gate" in names
    assert "readiness_gate" in names
    assert "bundle_staged_gate" in names


# --- the shell caller is a consumer of the parser too ----------------------


def test_the_acquisition_loop_s_launcher_call_parses(session):
    """The CALLER, parsed by the callee's own parser.

    This is the defect that was here: the launcher declared `--run-id`,
    `--gpu` and `--max-price` and nothing else, while `a3_acquire.sh` sends
    `--scr`, `--session-commit` and `--bundle` as well -- and `run_session`
    reads all three off the namespace. Every test in this file built its own
    namespace and called `spec(args)`, so the real command line never ran and
    argparse would have exited 2 on the loop's FIRST launcher invocation,
    after the capacity watch, the live re-price, the grant, a full launch-bound
    readiness sweep and the bundle had all completed.

    Read out of the shell source rather than restated, so editing the loop
    cannot leave this agreeing with a call nobody makes.
    """
    L, _, _ = session
    src = (REPO / "scripts/stages/stage-1/phase_a3/a3_acquire.sh").read_text()
    m = re.search(r"autoinit_a3_launch\.py(.*?)>\s*\"\$SCR/launcher\.log\"",
                  src, re.S)
    assert m, "the acquisition loop no longer invokes the A3 launcher"
    sent = re.findall(r"--[a-z][a-z0-9-]*", m.group(1))
    assert {"--scr", "--run-id", "--session-commit", "--bundle",
            "--gpu", "--max-price"} <= set(sent), sent

    parser = L.build_parser()
    known = {o for a in parser._actions for o in a.option_strings}
    unknown = sorted(set(sent) - known)
    assert not unknown, (
        f"a3_acquire.sh sends {unknown}, which the launcher's parser does not "
        f"define; argparse would exit 2 before any gate ran")

    #: And every REQUIRED flag is one the loop sends, which is the same exit 2
    #: from the other direction.
    required = {a.option_strings[0] for a in parser._actions
                if a.required and a.option_strings}
    assert not sorted(required - set(sent)), (
        f"the launcher requires {sorted(required - set(sent))}, which the "
        f"acquisition loop does not send")

    #: The namespace the loop's own argv produces must satisfy what
    #: `SessionRunner` reads off it. Enumerated from the runner's source so a
    #: new read cannot be missed here.
    runner = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    reads = {m.group(1) for m in re.finditer(r"\ba(?:rgs)?\.([a-z_]+)", runner)}
    argv = ["--scr", "/tmp/a3-contract", "--run-id", "a3_attempt1",
            "--session-commit", "0" * 40, "--bundle", "aad_test.bundle",
            "--gpu", "NVIDIA L40S", "--max-price", "1.09"]
    ns = parser.parse_args(argv)
    for attr in ("scr", "out", "runpod_config", "session_commit", "bundle"):
        assert attr in reads, (
            f"{attr} is no longer read by SessionRunner; this list is stale")
        assert hasattr(ns, attr), (
            f"the loop's argv produces no `{attr}`, which SessionRunner reads")


def test_the_evidence_path_the_loop_writes_is_the_one_the_aggregator_reads():
    """The handoff between the paid session and the `$0` comparison.

    `SessionRunner` fetches the archive into `<scr>/store` and extracts it
    into `<scr>/store/extracted`, so `--scr` decides where A3's only
    scientific output comes home to. The loop pointed `--scr` at a
    session-scoped temp directory and then printed an aggregation command
    naming `~/aad-artifacts/phase_a3/<run>` -- a path nothing wrote, one
    level off from the one that would have existed. Three trained and scored
    probes' evidence would have been in `/tmp`.

    Checked as a chain of three agreements, each read from its own owner:
    the loop's durable root, the archive's own top-level prefix, and the
    directory `aggregate_a3` opens.
    """
    loop = (REPO / "scripts/stages/stage-1/phase_a3/a3_acquire.sh").read_text()
    assert "ART=/home/ecs-user/aad-artifacts/phase_a3" in loop, (
        "the loop no longer declares a durable artifact root")
    assert 'BASE="$ART/_sessions"' in loop, (
        "the session scratch is not under the durable root, so "
        "<scr>/store/extracted is not durable")
    assert 'cp -a "$SCR/store/extracted/." "$EV/"' in loop, (
        "the loop does not place the EXTRACTED tree at the evidence root")
    assert "--evidence $EV" in loop, (
        "the printed aggregation command does not name the path the loop "
        "actually wrote")

    #: The extracted tree's top level is the artifact spec's prefix, and the
    #: aggregator opens `<evidence>/audit/autoinit_a3`.
    spec = json.loads(
        (REPO / "configs/autoinit/a3_artifacts.json").read_text())
    prefixes = {e["pattern"].split("/", 1)[0] for e in spec["entries"]}
    assert "audit" in prefixes, prefixes
    agg = (REPO / "scripts/stages/stage-1/phase_a3/aggregate_a3.py").read_text()
    assert 'evidence / "audit" / "autoinit_a3"' in agg, (
        "the aggregator no longer reads <evidence>/audit/autoinit_a3; the "
        "loop's copy would land somewhere it does not look")


def test_the_run_ids_the_loop_mints_are_run_ids_the_layout_accepts(session):
    """The shell names the run; the layout decides what a run may be called.

    `a3_attempt1` and `a3-attempt2` produced correct-looking paths through
    `rel_run_dir`, every `$0` gate, four governance artifacts and a staged
    22.8 MB bundle -- and `open_run` refused the hyphen at the last step of
    the chain, twice, because `RunLayout` accepts only `[a-z0-9][a-z0-9_]*`
    and `rel_run_dir` interpolates without checking. Two owners of one string
    and only the later one looks.

    So the spelling the loop mints is checked against the owner's rule, and
    the launcher's parser refuses a bad one before the grant is written.
    """
    L, _, _ = session
    loop = (REPO / "scripts/stages/stage-1/phase_a3/a3_acquire.sh").read_text()
    m = re.search(r'^\s*RUN="([^"]+)"', loop, re.M)
    assert m, "the loop no longer mints a run name"
    minted = m.group(1).replace("$N", "7")
    from aadistill.runtime.run_layout import _ID

    assert _ID.match(minted), (
        f"the loop mints {minted!r}, which RunLayout refuses; open_run would "
        f"abort after the whole one-use chain had been built")

    #: And the parser refuses the shape that was actually used, at parse time.
    for bad in ("a3_bad-attempt7", "A3attempt7", "../escape"):
        with pytest.raises(SystemExit):
            L.build_parser().parse_args(
                ["--scr", "/tmp/a3-contract", "--run-id", bad,
                 "--session-commit", "0" * 40, "--bundle", "b.bundle"])
    ok = L.build_parser().parse_args(
        ["--scr", "/tmp/a3-contract", "--run-id", minted,
         "--session-commit", "0" * 40, "--bundle", "b.bundle"])
    assert ok.run_id == minted

    #: The loop must also count BOTH spellings when picking the next number,
    #: or it would reuse an attempt number that already has evidence on disk.
    assert 'a3-attempt$N' in loop and 'a3_attempt$N' in loop, (
        "next_free does not consider both spellings; an attempt number with "
        "a consumed chain under the old name would be reused")


def test_the_card_is_part_of_the_measurement_not_only_of_the_price(session):
    """A3 may not run on an approved-but-different architecture.

    The approved tier holds three types and `require_approved` accepts any of
    them, which is correct for a session whose success is a measurement and
    wrong for one whose success is a DIGEST: stage E compares A-bsz1's
    artifact byte-exactly against an incumbent built on an L40S, and a GEMM
    reduces in an architecture-dependent order. A different card can fail
    that gate for a reason that has nothing to do with the batching protocol
    -- an integrity stop on hardware, reported as a protocol failure, after a
    paid session.

    This was reachable: RTX 6000 Ada was `usable` at $0.84 while the L40S
    was refusing on capacity, so the acquisition loop's watch would have
    handed the launcher a cheaper, approved, scientifically void card.
    """
    L, args, _ = session
    import json as _json

    priced = _json.loads((REPO / "logs/stages/stage-1/phase_c3/plans/"
                          "a3_live_pricing.json").read_text())["gpu"]
    auth = types.SimpleNamespace(hard_cap_usd=L.a3_hard_ceiling_usd(REPO),
                                 harness_source_digest="x" * 64,
                                 harness_source_files=())

    def _gate(gpu, rate):
        a = L.build_parser().parse_args(
            ["--scr", "/tmp/a3-contract", "--run-id", "a3_attempt1",
             "--session-commit", "0" * 40, "--bundle", "b.bundle",
             "--gpu", gpu, "--max-price", str(rate)])
        a.disk_gb = 60
        return L.pricing_identity_gate(_ctx(a, auth=auth))

    ok, why = _gate(priced, 1.09)
    assert ok, why
    for other in ("NVIDIA RTX 6000 Ada Generation", "NVIDIA L40"):
        if other == priced:
            continue
        ok, why = _gate(other, 0.84)
        assert not ok, f"{other} was accepted; the digest gate is architecture-bound"
        assert "byte-exact digest" in why

    #: And the acquisition loop's capacity watch offers only that card, so a
    #: chain is not consumed to discover the refusal.
    loop = (REPO / "scripts/stages/stage-1/phase_a3/a3_acquire.sh").read_text()
    assert 'want = priced["gpu"]' in loop, (
        "the watch no longer filters to the priced card")
    assert "select(query_offers())" not in loop, (
        "the watch still takes the best AVAILABLE approved type")

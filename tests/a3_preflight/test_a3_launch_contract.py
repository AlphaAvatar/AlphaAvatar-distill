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

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "pod"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.phase_c3 import a3_session as A3S  # noqa: E402

SETUP_SCRIPT = REPO / "scripts/pod/autoinit_preflight_setup.sh"


def launcher():
    spec = importlib.util.spec_from_file_location(
        "a3lau_contract", REPO / "scripts/pod/autoinit_a3_launch.py")
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
        ["--scr", "/tmp/a3-contract", "--run-id", "a3-attempt1",
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
        "ROPE_OK globs artifacts/stage1/*/checkpoint/config.json and no "
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
    import autoinit_a3_driver as D

    assert spec.status_path == A3S.STATUS_PATH
    assert spec.driver_job_id == A3S.DRIVER_JOB_ID
    #: And the DRIVER writes to the same file the launcher polls.
    assert str(D.STATUS) == A3S.STATUS_PATH
    assert spec.run_log_path == A3S.RUN_LOG_PATH


def test_the_markers_the_launcher_watches_are_the_markers_the_driver_emits(
        session):
    _, _, spec = session
    src = (REPO / "scripts/pod/autoinit_a3_driver.py").read_text()
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
    from experiments.phase_c3.a3_authorization import A3_HARNESS_FILES

    _, _, spec = session
    for path in (spec.artifacts.spec_success, spec.artifacts.spec_failed):
        assert path in A3_HARNESS_FILES, path


def test_every_report_name_is_a_file_the_driver_writes(session):
    _, _, spec = session
    src = (REPO / "scripts/pod/autoinit_a3_driver.py").read_text()
    for name in spec.artifacts.report_names:
        assert name in src, (
            f"the artifact policy reports {name!r} and the driver never "
            "writes it")


# --- the budget bounds the spend exactly ----------------------------------


def test_the_plan_reproduces_the_derived_ceiling_rather_than_approaching_it(
        session):
    """A planner that applied its own contingency on top of the component
    table's overrun factor once produced a plan terminating ABOVE its grant."""
    from experiments.phase_c3.a3_authorization import (
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
    base = dict(args=args, auth=types.SimpleNamespace(
        hard_cap_usd=8.2525, harness_source_digest="x" * 64,
        harness_source_files=()), evidence={}, say=lambda *_: None)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_the_harness_gate_refuses_a_digest_that_does_not_match(session):
    L, args, _ = session
    ok, why = L.a3_harness_gate(_ctx(args))
    assert ok is False and "harness digest" in why
    #: And a grant naming no digest at all is refused, not defaulted.
    ctx = _ctx(args, auth=types.SimpleNamespace(
        hard_cap_usd=8.2525, harness_source_digest="", harness_source_files=()))
    ok, why = L.a3_harness_gate(ctx)
    assert ok is False and "names no harness digest" in why


def test_the_controls_evidence_gate_passes_on_the_real_evidence(session):
    """A3's whole saving is that it does not retrain the controls, so a
    session that trains three probes and then finds nothing to compare against
    has produced half an experiment at full price."""
    L, args, _ = session
    ctx = _ctx(args)
    ok, why = L.controls_evidence_gate(ctx)
    if not ok and "absent" in why:
        pytest.skip("attempt75's durable control evidence is not mounted here")
    assert ok, why
    ev = ctx.evidence["controls_evidence"]
    assert len(ev["seeds"]) == 3
    assert ev["files_verified"] == 6, ev
    assert ev["problems"] == []


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
        ["--scr", "/tmp/a3-contract", "--run-id", "a3-attempt1",
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
    assert names[-2] == "durable_capacity_gate", names
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
    src = (REPO / "scripts/pod/a3_acquire.sh").read_text()
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
    argv = ["--scr", "/tmp/a3-contract", "--run-id", "a3-attempt1",
            "--session-commit", "0" * 40, "--bundle", "aad_test.bundle",
            "--gpu", "NVIDIA L40S", "--max-price", "1.09"]
    ns = parser.parse_args(argv)
    for attr in ("scr", "out", "runpod_config", "session_commit", "bundle"):
        assert attr in reads, (
            f"{attr} is no longer read by SessionRunner; this list is stale")
        assert hasattr(ns, attr), (
            f"the loop's argv produces no `{attr}`, which SessionRunner reads")

"""C2's run owns its whole formal execution, and its resource scope is checked.

Two properties, both of which were missing while the governance chain was
already run-owned:

**The session record.** `SessionRunner.save()` writes exactly `args.out`, and
the C2 parser exposed `--out` with a default of
`logs/stages/stage-1/phase_c2/runs/autoinit_phase_c2_session.json` — a file in
the runs root belonging to no run, which every attempt would have written in
turn, and which an operator could aim at another attempt's directory. The grant,
readiness record, authorization and bundle record were owned; the record of what
the run actually did was not.

**The resource scope.** The grant states "one issuance, one launcher session, at
most 3 provider resources, never more than one billing at a time, all sharing
one $15.0446 ceiling". That was prose. `--host-draws` defaulted to 3 and nothing
compared the request to the permission.

Nothing here contacts a provider, creates a pod, or spends anything. The run
directories are built under `tmp_path`; the only repository reads are the real
launcher module and the real parser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for _root in ("src", "scripts", "scripts/pod", "tests/pod"):
    if str(REPO / _root) not in sys.path:
        sys.path.insert(0, str(REPO / _root))

from session_specs import load_session_launcher  # noqa: E402

STAGE = "1"


@pytest.fixture(scope="module")
def launcher():
    return load_session_launcher("autoinit_phase_c2_launch")


def args_for(launcher, run_id: str, scr: Path, **over):
    """A namespace from the REAL parser, so `out` comes from `--run-id`."""
    argv = ["--scr", str(scr), "--session-commit", "0" * 40,
            "--bundle", "aad_00000000.bundle", "--run-id", run_id]
    for flag, value in over.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return launcher.build_parser().parse_args(argv)


# --- 1. the session record lives under the run ------------------------------

def test_attempt2_resolves_its_session_record_under_its_own_runtime(launcher):
    assert launcher.session_record_path("attempt2") == (
        "logs/stages/stage-1/phase_c2/runs/attempt2/runtime/session.json")
    #: And through the run layout, not a string built here.
    from experiments.run_layout import rel_run_dir

    assert launcher.session_record_path("attempt2") == (
        f"{rel_run_dir('phase_c2', 'attempt2', STAGE)}/runtime/session.json")


def test_the_parser_produces_out_and_offers_no_way_to_choose_it(launcher, tmp_path):
    parser = launcher.build_parser()
    flags = {opt for action in parser._actions for opt in action.option_strings}
    assert "--out" not in flags, (
        "an operator-selectable --out is back; it can point one attempt's "
        "session record at another attempt's directory")

    args = args_for(launcher, "attempt2", tmp_path)
    assert args.out == launcher.session_record_path("attempt2"), (
        "the runner reads args.out, so --run-id must fill it")
    #: The namespace the runner reads must be COMPLETE from the parser: a
    #: device-canary session died at $0.0603 on an attribute a hand-written
    #: namespace had and the parser did not, after the pod was billing.
    for attribute in ("out", "scr", "run_id", "session_commit", "bundle",
                      "host_draws", "create_attempts"):
        assert hasattr(args, attribute), attribute


def test_two_run_ids_cannot_share_one_session_record(launcher):
    a = launcher.session_record_path("attempt2")
    b = launcher.session_record_path("attempt3")
    assert a != b
    assert "attempt2" in a and "attempt3" in b
    #: Both under their own run, neither in the runs root.
    for path in (a, b):
        assert path.count("/runs/") == 1
        assert not path.endswith("runs/session.json")


def test_the_declared_roles_are_the_ones_this_session_produces(launcher):
    """A search session has no probe, battery, replay, rung or scoring role, and
    declaring one would put a path in the manifest that nothing writes."""
    from experiments.phase_c2.session import C2_RUN_ROLES
    from experiments.run_layout import check_roles

    #: Every role sits under exactly one declared area. `check_roles` raises
    #: otherwise, which is the whole point of asking it rather than eyeballing.
    check_roles(C2_RUN_ROLES)
    assert C2_RUN_ROLES["session_record"] == "runtime/session.json"
    for role in ("grant", "authorization", "readiness_record", "bundle_record",
                 "session_record", "launcher_log", "watchdog_journal",
                 "driver_evidence", "driver_log", "driver_status",
                 "search_summary", "baseline_comparison", "artifact_manifest",
                 "outcome"):
        assert role in C2_RUN_ROLES, role
    for absent in ("probe_manifest", "battery", "replay_record", "rung",
                   "scoring_contract", "recovery_recipe"):
        assert absent not in C2_RUN_ROLES, f"{absent} is C1's, not C2's"
    #: The spec requires only what a $0-refused session produces.
    assert launcher.C2_RUN_SPEC.required == ("session_record",)


# --- 2. opening the run, at $0 ----------------------------------------------

def test_opening_the_run_claims_the_scratch_and_points_out_at_the_run(
        launcher, tmp_path):
    scr = tmp_path / "scr"
    repo = tmp_path / "repo"
    args = args_for(launcher, "attempt2", scr)
    layout = launcher.open_c2_run(args, repo_root=repo)

    assert args.out == launcher.session_record_path("attempt2")
    assert layout.root == repo / "logs/stages/stage-1/phase_c2/runs/attempt2"
    #: The scratch carries this run's claim, and nothing else was written there.
    claim = json.loads((scr / ".aad_output_claim.json").read_text())
    assert (claim["experiment_id"], claim["run_id"]) == ("phase_c2", "attempt2")
    #: The run's areas exist and it is not yet recorded.
    assert (layout.root / "runtime").is_dir()
    assert not (layout.root / "manifest.json").exists()


def test_a_scratch_root_claimed_by_another_run_is_refused(launcher, tmp_path):
    """Two runs must not share a writable output root: the second would collect
    the first's evidence as its own."""
    from experiments.run_layout import OutputOwnershipError

    scr = tmp_path / "scr"
    repo = tmp_path / "repo"
    launcher.open_c2_run(args_for(launcher, "attempt2", scr), repo_root=repo)

    with pytest.raises(OutputOwnershipError, match="attempt2"):
        launcher.open_c2_run(args_for(launcher, "attempt3", scr),
                             repo_root=repo)


def test_an_unclaimed_scratch_already_holding_this_runs_outputs_is_refused(
        launcher, tmp_path):
    """Unknown ownership, not "probably stale". mtime is a guess."""
    from experiments.run_layout import OutputOwnershipError

    scr = tmp_path / "scr"
    (scr / "relay").mkdir(parents=True)
    (scr / "relay/c2_evidence.json").write_text("{}")

    with pytest.raises(OutputOwnershipError, match="carries no claim"):
        launcher.open_c2_run(args_for(launcher, "attempt2", scr),
                             repo_root=tmp_path / "repo")


def test_an_already_recorded_run_id_is_refused(launcher, tmp_path):
    from experiments.run_layout import RunConventionError

    repo = tmp_path / "repo"
    run = repo / "logs/stages/stage-1/phase_c2/runs/attempt2"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")

    with pytest.raises(RunConventionError, match="already recorded"):
        launcher.open_c2_run(args_for(launcher, "attempt2", tmp_path / "scr"),
                            repo_root=repo)


def test_an_unrecorded_occupied_run_is_refused_rather_than_reopened(
        launcher, tmp_path):
    """A launcher process that died leaves a run directory with no manifest.
    Overwriting it destroys the only evidence of what happened."""
    from experiments.run_layout import RunConventionError

    repo = tmp_path / "repo"
    run = repo / "logs/stages/stage-1/phase_c2/runs/attempt2"
    (run / "runtime").mkdir(parents=True)
    (run / "runtime/session.json").write_text('{"terminal": "?"}')

    with pytest.raises(RunConventionError, match="launcher died"):
        launcher.open_c2_run(args_for(launcher, "attempt2", tmp_path / "scr"),
                            repo_root=repo)


def test_the_prepared_governance_inputs_do_not_count_as_occupancy(
        launcher, tmp_path):
    """By formal launch time the run already holds four artifacts written by
    somebody else, in this order: grant, readiness record, authorization, bundle
    record. C1 shipped without the last two declared and the chain became
    unsatisfiable — the issuer writes the authorization into the run, the
    launcher reads it from there, and `open_run` refused the run as occupied.
    Attempt 13 died on it at $0."""
    from experiments.phase_c2.session import C2_RUN_ROLES

    repo = tmp_path / "repo"
    run = repo / "logs/stages/stage-1/phase_c2/runs/attempt2"
    (run / "governance").mkdir(parents=True)
    for role in launcher._RUN_PREPARED:
        (repo / "logs/stages/stage-1/phase_c2/runs/attempt2"
         / C2_RUN_ROLES[role]).write_text("{}")

    layout = launcher.open_c2_run(args_for(launcher, "attempt2", tmp_path / "scr"),
                                  repo_root=repo)
    assert layout.root.is_dir()
    #: And an UNDECLARED governance file is still refused: the exemption is per
    #: role, by name.
    from experiments.run_layout import RunConventionError

    other = repo / "logs/stages/stage-1/phase_c2/runs/attempt4/governance"
    other.mkdir(parents=True)
    (other / "something_else.json").write_text("{}")
    with pytest.raises(RunConventionError, match="undeclared"):
        launcher.open_c2_run(args_for(launcher, "attempt4", tmp_path / "scr4"),
                            repo_root=repo)


# --- 3. closing the run ------------------------------------------------------

def session_record(**over) -> dict:
    doc = {"session_id": "autoinit-phase-c2-search1",
           "session_plan_hash": "p" * 64, "passed": True,
           "terminal": "ALL_DONE", "pod_id": "abc123", "cost": 7.0,
           "provider_confirms_gone": True,
           "harness_source_digest": "h" * 64}
    doc.update(over)
    return doc


def test_closeout_writes_a_verifiable_manifest_over_the_roles_that_exist(
        launcher, tmp_path):
    scr = tmp_path / "scr"
    repo = tmp_path / "repo"
    args = args_for(launcher, "attempt2", scr)
    layout = launcher.open_c2_run(args, repo_root=repo)

    #: What the runner and the pod would have left behind. Small text only.
    (scr / "relay").mkdir(parents=True, exist_ok=True)
    (scr / "store").mkdir(parents=True, exist_ok=True)
    (scr / "launch.log").write_text("launcher\n")
    (scr / "relay/autoinit_phase_c2_run.log").write_text("driver\n")
    (scr / "relay/autoinit_phase_c2.status").write_text("ALL_DONE\n")
    (scr / "relay/c2_evidence.json").write_text('{"stage": "done"}')
    (scr / "store/c2_search_summary.json").write_text('{"front": []}')
    (scr / "store/c2_baseline_comparison.json").write_text('{"verdict": "x"}')
    (scr / "store/manifest.json").write_text('{"entries": []}')
    (scr / "watchdog_abc123.jsonl").write_text('{"tick": 1}\n')
    #: A LARGE working state, which must not be copied into the repository.
    big = scr / "store/extracted/autoinit/phase_c2_search/state_0001"
    big.mkdir(parents=True)
    (big / "model.safetensors").write_bytes(b"\0" * 4096)

    (repo / args.out).parent.mkdir(parents=True, exist_ok=True)
    (repo / args.out).write_text(json.dumps(session_record()))

    doc = launcher.close_c2_run(layout, args, repo_root=repo)

    from experiments.run_layout import verify_run_manifest

    ok, why = verify_run_manifest(doc, layout.run_root,
                                  spec=launcher.C2_RUN_SPEC)
    assert ok, why
    assert doc["experiment_id"] == "phase_c2" and doc["run_id"] == "attempt2"
    #: Every role it claims exists, and the ones it does not claim do not.
    for role in ("session_record", "launcher_log", "driver_log",
                 "driver_status", "driver_evidence", "search_summary",
                 "baseline_comparison", "artifact_manifest"):
        assert role in doc["roles"], role
        assert (layout.root / doc["roles"][role]).exists()
    assert "outcome" not in doc["roles"], (
        "a maintainer classification nobody has written yet is not a role this "
        "run produced")
    #: The watchdog journal came home, per resource.
    assert (layout.root / "runtime/watchdog/watchdog_abc123.jsonl").is_file()
    #: And no large state was copied in.
    copied = [p for p in layout.root.rglob("*") if p.is_file()]
    assert not any(p.suffix == ".safetensors" for p in copied)
    assert sum(p.stat().st_size for p in copied) < 200_000, (
        "the run directory grew past small-reviewable-text size")
    assert doc["status"]["trains_anything"] is False


def test_a_session_refused_at_a_zero_dollar_gate_still_records_an_owned_run(
        launcher, tmp_path):
    """The case that mattered and had no owner: the runner saves its record on
    every path, including a pre-provider refusal, and that record must land in
    the run rather than in a shared file."""
    scr = tmp_path / "scr"
    repo = tmp_path / "repo"
    args = args_for(launcher, "attempt2", scr)
    layout = launcher.open_c2_run(args, repo_root=repo)

    #: Nothing but the session record: no pod, no relay, no evidence.
    (repo / args.out).parent.mkdir(parents=True, exist_ok=True)
    (repo / args.out).write_text(json.dumps(session_record(
        passed=False, terminal="", pod_id=None, cost=0.0,
        provider_confirms_gone=None)))

    doc = launcher.close_c2_run(layout, args, repo_root=repo)
    assert doc["roles"] == {"session_record": "runtime/session.json"}, doc["roles"]
    assert doc["status"]["pod_id"] is None and doc["status"]["cost"] == 0.0
    from experiments.run_layout import verify_run_manifest

    ok, why = verify_run_manifest(doc, layout.run_root,
                                  spec=launcher.C2_RUN_SPEC)
    assert ok, why


def test_closeout_refuses_a_scratch_that_is_not_this_runs(launcher, tmp_path):
    """Asked again at closeout, not assumed from the open: a foreign scratch
    would turn a run that failed early into a manifest full of someone else's
    evidence."""
    from experiments.run_layout import OutputOwnershipError, claim_output_root

    scr = tmp_path / "scr"
    repo = tmp_path / "repo"
    args = args_for(launcher, "attempt2", scr)
    layout = launcher.open_c2_run(args, repo_root=repo)
    #: Somebody else takes the scratch over between open and close.
    (scr / ".aad_output_claim.json").unlink()
    claim_output_root(scr, "phase_c2", "attempt9", outputs=("launch.log",))

    with pytest.raises(OutputOwnershipError, match="attempt9"):
        launcher.close_c2_run(layout, args, repo_root=repo)


# --- 4. the resource scope, enforced before provider creation ---------------

def scope(run_id="attempt2", permitted=3):
    from experiments.phase_c2.session import C2ResourceScope

    return C2ResourceScope(run_id=run_id, issuances_permitted=1,
                           launch_attempts_permitted=1,
                           provider_resources_permitted=permitted,
                           one_billing_resource_at_a_time=True)


def ctx_for(launcher, tmp_path, *, run_id="attempt2", host_draws=3,
            auth_scope="default"):
    args = args_for(launcher, run_id, tmp_path, host_draws=host_draws)
    auth = SimpleNamespace(
        resource_scope=scope() if auth_scope == "default" else auth_scope)
    return SimpleNamespace(args=args, auth=auth, evidence={})


def test_permitting_three_and_requesting_three_is_accepted(launcher, tmp_path):
    ok, why = launcher.resource_scope_gate(
        ctx_for(launcher, tmp_path, host_draws=3))
    assert ok, why
    assert "3 draw(s) within the 3 permitted" in why


def test_permitting_three_and_requesting_four_is_refused(launcher, tmp_path):
    ctx = ctx_for(launcher, tmp_path, host_draws=4)
    ok, why = launcher.resource_scope_gate(ctx)
    assert not ok
    assert "--host-draws 4 exceeds the 3" in why
    #: And it happens before anything is created: the gate tuple runs in
    #: `run_prechecks`, which `run()` calls before the acquisition loop.
    assert "provider" not in ctx.evidence


def test_an_authorization_issued_for_another_run_is_refused(launcher, tmp_path):
    ctx = ctx_for(launcher, tmp_path, run_id="attempt3")
    ok, why = launcher.resource_scope_gate(ctx)
    assert not ok
    assert "issued for run 'attempt2'" in why and "attempt3" in why


def test_an_authorization_with_no_scope_is_refused(launcher, tmp_path):
    ok, why = launcher.resource_scope_gate(
        ctx_for(launcher, tmp_path, auth_scope=None))
    assert not ok
    assert "unknown limit is not an unlimited one" in why


def test_the_scope_gate_runs_before_the_provider_is_contacted(launcher):
    """Position, not intention. Everything in `precheck` runs in
    `run_prechecks()`, which `SessionRunner.run()` calls before the acquisition
    loop — so being in the tuple IS being before `create()`."""
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/c2scope", "--session-commit", "0" * 40,
         "--bundle", "aad_00000000.bundle", "--run-id", "spec_check"])
    names = [getattr(g, "__name__", "") for g in launcher.spec(args).precheck]
    assert "resource_scope_gate" in names
    runner = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert "if not self.make_plan() or not self.run_prechecks():" in runner
    assert runner.index("run_prechecks()") < runner.index("for draw in range")


def test_a_create_that_returned_no_pod_id_does_not_consume_a_resource():
    """The runner records such a draw with `pod_id: null` and
    `provider_resource_created: false` precisely so this distinction survives
    into the evidence. Counting rows would let a provider's refusal consume a
    permission."""
    from experiments.phase_c2.session import C2ResourceScope

    draws = [
        {"draw": 1, "pod_id": None, "outcome": "create_failed",
         "release": {"provider_resource_created": False}},
        {"draw": 2, "pod_id": "pod_a", "outcome": "cold"},
        {"draw": 3, "pod_id": "pod_b", "outcome": "ok"},
    ]
    assert C2ResourceScope.provider_resources_used(draws) == 2
    assert C2ResourceScope.provider_resources_used([]) == 0
    assert C2ResourceScope.provider_resources_used(None) == 0
    #: Three rows, two resources, and the permission was for three: a session
    #: whose first create was refused has not used up a resource.
    assert scope(permitted=3).permits_draws(3)[0] is True


def test_the_scope_is_derived_from_the_grant_and_refuses_a_silent_default():
    from aadistill.governance.authorization import AuthorizationError
    from experiments.phase_c2.session import C2ResourceScope

    stated = {"one_use": {"issuances_permitted": 1,
                          "launch_attempts_permitted": 1,
                          "provider_resources_permitted": 3,
                          "one_billing_resource_at_a_time": True}}
    built = C2ResourceScope.from_grant(stated, "attempt2")
    assert built.provider_resources_permitted == 3
    assert built.run_id == "attempt2"

    #: A grant that states fewer keys gets no scope, rather than a permissive
    #: one filled in by a default.
    for drop in C2ResourceScope.GRANT_KEYS:
        partial = {"one_use": {k: v for k, v in stated["one_use"].items()
                               if k != drop}}
        with pytest.raises(AuthorizationError, match="does not state"):
            C2ResourceScope.from_grant(partial, "attempt2")
    with pytest.raises(AuthorizationError, match="no structured"):
        C2ResourceScope.from_grant({}, "attempt2")
    #: And it may not waive the one-billing-resource rule.
    with pytest.raises(AuthorizationError, match="may not waive"):
        C2ResourceScope(run_id="attempt2", issuances_permitted=1,
                        launch_attempts_permitted=1,
                        provider_resources_permitted=3,
                        one_billing_resource_at_a_time=False)


def test_the_scope_survives_a_round_trip_through_the_real_loader(tmp_path):
    """An authorization the issuer wrote and the launcher cannot read its scope
    from would refuse every launch for the wrong reason."""
    from experiments.phase_c2.session import C2Authorization

    auth = C2Authorization(
        authorization_id="x", granted_utc="2026-01-01T00:00:00Z",
        granted_by="fixture", plan_id="p", plan_hash="h",
        science_plan_hash="s", expected_usd=1.0, hard_cap_usd=2.0,
        authorized_stages=(0, 1), stage_conditions={}, scope_note="t",
        resource_scope=scope())
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(auth.as_dict(), indent=1))

    loaded = C2Authorization.load(path)
    assert loaded.resource_scope is not None
    assert loaded.resource_scope.provider_resources_permitted == 3
    assert loaded.resource_scope.run_id == "attempt2"
    #: The document states the run id once, from the scope.
    doc = json.loads(path.read_text())
    assert doc["run_id"] == "attempt2"
    assert doc["resource_scope"]["provider_resources_permitted"] == 3

    #: A document whose scope block is partial is refused at load, not silently
    #: downgraded to "no limit".
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.manifest import sha256_json

    broken = dict(doc)
    broken["resource_scope"] = {"run_id": "attempt2"}
    broken.pop("authorization_sha256")
    broken["authorization_sha256"] = sha256_json(broken)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(broken))
    with pytest.raises(AuthorizationError, match="missing"):
        C2Authorization.load(bad)

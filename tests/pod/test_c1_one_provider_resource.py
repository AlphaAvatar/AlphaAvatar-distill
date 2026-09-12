"""At most ONE BILLING resource at any instant. The launcher must be unable to
have two.

**Superseded, prospectively, 2026-09-11.** The rule this module was written for
was "one provider resource per session, ever". Attempt 11 showed the cost of
that: its pod was created, billed, and never became reachable -- an ordinary
provider cold host -- and because `host_draws` was pinned at 1, a condition
every other session in this project handles by taking another draw consumed a
whole formal attempt instead.

Draws are permitted again, capped at a batch of three. What is NOT relaxed is
the property the old rule was really protecting, and it is now enforced where it
belongs: `SessionRunner.release_and_confirm` will not let the next resource be
created until the PROVIDER reports the abandoned one not billing. The previous
arrangement made the redraw branch unreachable, and a dead branch is a poor
guard -- that one had been clearing `pod_id` locally and continuing, treating a
subprocess returning and a variable being assigned as evidence that a pod had
stopped costing money.

The grant says it plainly: one issuance, one launch attempt, one provider
resource, and "after consumption there is NO retry and NO replacement pod". The
launcher did not enforce any of it. It defaulted to `--create-attempts 8` and
`--host-draws 3`, so:

* a cold or endpoint-less first pod was deleted and a SECOND one drawn — a
  replacement pod, which is exactly the thing the grant forbids;
* a create failure slept `--create-retry-seconds` (300 s) and tried again, up to
  seven more times, waiting on the stock the grant says not to chase.

Neither would have been visible in a launch transcript as a violation; both look
like ordinary resilience. That is why this is enforced by TYPE and by spec
construction rather than by a default a future launch command has to remember.

`SessionRunner` is untouched. Phase A, Phase B, both continuations and the
preflight keep multi-draw acquisition, which is correct for them: they have no
one-resource grant.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import types

from aadistill.infrastructure.session import ExecutionCommands  # noqa: E402
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))

from aadistill.infrastructure import session_runner as SR  # noqa: E402
from aadistill.infrastructure.provider import PodState  # noqa: E402

LAUNCHER = REPO / "scripts/pod/autoinit_c1_launch.py"


def _launcher():
    spec = importlib.util.spec_from_file_location("c1_launch_one", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["c1_launch_one"] = mod
    spec.loader.exec_module(mod)
    return mod


C1 = _launcher()
#: `--run-id` is required by the launcher: a run is named before it is launched,
#: and no test builds an argv the operator could not type. Nothing here opens a
#: run — these tests read `create_attempts`/`host_draws` off the parsed namespace
#: and drive `SessionRunner` directly — so the id names no directory.
BASE_ARGV = ["--scr", "/tmp/c1-one-resource", "--session-commit", "0" * 40,
             "--bundle", "aad_test.bundle", "--run-id", "one_resource"]


# --- A. the defaults realize one resource -----------------------------------

def test_A_the_c1_defaults_are_one_create_attempt_and_a_batch_of_draws():
    args = C1.build_parser().parse_args(BASE_ARGV)
    #: A create-attempt sleeps and asks the same market again. A draw replaces
    #: an unusable host. Only the second is authorized.
    assert args.create_attempts == 1
    assert args.host_draws == C1.C1_MAX_HOST_DRAWS == 3
    assert C1.C1_CREATE_ATTEMPTS == 1


def test_A2_the_spec_realizes_them_too():
    """The value the RUNNER will read, not just the one the parser produced."""
    args = C1.build_parser().parse_args(BASE_ARGV)
    C1.spec(args)                       # must not raise
    assert args.create_attempts == 1 and args.host_draws == 3


# --- B. anything else is refused --------------------------------------------

@pytest.mark.parametrize("flag,value", [
    ("--create-attempts", "2"), ("--create-attempts", "8"),
    ("--host-draws", "4"), ("--host-draws", "8"), ("--host-draws", "0"),
])
def test_B_the_parser_refuses_values_outside_the_range(flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        C1.build_parser().parse_args([*BASE_ARGV, flag, value])
    assert exc.value.code == 2
    assert "outside" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [("create_attempts", 3),
                                         ("host_draws", 9),
                                         ("host_draws", 0)])
def test_B2_spec_construction_refuses_a_hand_built_namespace(field, value):
    """A namespace is not a parser. Device-canary attempt 1 died at $0.0603 on
    exactly that difference, so the rule is enforced in both places."""
    args = C1.build_parser().parse_args(BASE_ARGV)
    setattr(args, field, value)
    with pytest.raises(SystemExit, match="refusing to build the C1 session"):
        C1.spec(args)


# --- C/D/E. what the runner actually does with those values -----------------

class _CountingProvider:
    """Models the control plane, including that a removed pod stops billing.

    `releases` is the knob the new guard is about: when it is False the provider
    keeps reporting the pod as billing no matter how often it is asked, which is
    what an unconfirmed release looks like from inside the runner.
    """

    def __init__(self, releases: bool = True):
        self.terminated: list[str] = []
        self.releases = releases

    def get(self, pod_id):
        gone = self.releases and pod_id in self.terminated
        return PodState(pod_id=pod_id, exists=not gone,
                        desired_status="TERMINATED" if gone else "RUNNING")

    def terminate(self, pod_id):
        self.terminated.append(pod_id)
        return []


def _runner(monkeypatch, *, outcome, create_ok=True, releases=True,
            host_draws=None):
    """The REAL `SessionRunner.run` acquisition loop, with only the provider
    boundary faked. Everything between `create()` and the draw decision is
    production code."""
    r = object.__new__(SR.SessionRunner)
    r.provider = _CountingProvider(releases=releases)
    r.cli = "runpodctl"
    r.pod_id = ""
    r.price = 1.09
    r.start_epoch = 0.0
    r.ev = {}
    r.scr = Path("/tmp/c1-one-resource-scr")
    r.scr.mkdir(parents=True, exist_ok=True)
    r.say = lambda m: r.ev.setdefault("said", []).append(m)
    r.save = lambda: None
    r.make_plan = lambda: True
    r.run_prechecks = lambda: True
    r._watchdog_for = ""
    r.launch_watchdog = lambda: Path("/dev/null")
    r.teardown_now = lambda why: r.ev.setdefault("teardown", []).append(why)
    #: NOT stubbed: `release_and_confirm`, `record_draw` and
    #: `_watchdog_journal_name` are the production code under test here.
    r.plan = types.SimpleNamespace(hard_terminate_minutes=834.0)
    r.spec = types.SimpleNamespace(        #: The runner reads its executables from the spec now, so a stub spec
        #: must declare them; there is no default for it to fall back on.
        #: The REAL type, not a SimpleNamespace. A fake that names the fields
        #: it happens to need goes stale silently the moment the type gains
        #: one -- which is exactly what happened when `workspace_root`,
        #: `checkout_root` and `min_cuda_version` were added: twenty tests
        #: failed with AttributeError on a double, not on the code.
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            remote_python="/opt/train/bin/python",
            workspace_root="/workspace", checkout_root="/workspace/aad",
            min_cuda_version="13.0"),
        teardown=types.SimpleNamespace(require_provider_confirmation=True),
        session_id="autoinit-c1")

    args = C1.build_parser().parse_args(BASE_ARGV)
    r.a = types.SimpleNamespace(
        create_attempts=args.create_attempts,
        host_draws=args.host_draws if host_draws is None else host_draws,
        create_retry_seconds=args.create_retry_seconds,
        gpu="NVIDIA L40S", max_price=1.09, image="img", disk_gb=200)

    created: list[int] = []

    def fake_run(cmd, **kw):
        if "create" in cmd:
            created.append(1)
            out = ('{"id":"pod%d","costPerHr":1.09}' % len(created)
                   if create_ok else "error: no capacity")
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if "remove" in cmd:
            r.provider.terminated.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(SR.subprocess, "run", fake_run)
    monkeypatch.setattr(SR.time, "sleep",
                        lambda s: r.ev.setdefault("slept", []).append(s))
    monkeypatch.setattr(SR.SessionRunner, "setup_on_draw",
                        lambda self, draw: outcome)
    return r, created


def test_C_a_cold_resource_is_RELEASED_WITH_CONFIRMATION_then_redrawn(monkeypatch):
    """The behaviour attempt 11 could not have: a cold host costs a draw, not
    the session."""
    r, created = _runner(monkeypatch, outcome="cold")

    assert r.run() is False
    assert len(created) == 3, f"expected three draws, got {len(created)}"
    said = " ".join(r.ev.get("said", []))
    assert "redrawing" in said
    assert not r.ev.get("slept_for_stock"), "C1 must not sleep against stock"


def test_D_a_no_endpoint_resource_does_the_same(monkeypatch):
    """Attempt 11's exact failure mode, by name."""
    r, created = _runner(monkeypatch, outcome="no_endpoint")

    assert r.run() is False
    assert len(created) == 3
    assert [d["outcome"] for d in r.ev["draws"]] == ["no_endpoint"] * 2, (
        "each abandoned draw must record its own outcome")


def test_E_the_next_resource_is_not_created_until_release_is_CONFIRMED(monkeypatch):
    """The guard, and the whole reason the redraw branch may exist again.

    A provider that keeps reporting the pod as billing must stop the session
    dead. The old branch fired `remove`, ignored the result, cleared `pod_id`
    and created the next pod -- two billing resources, and the second watchdog
    watching only the second.
    """
    r, created = _runner(monkeypatch, outcome="no_endpoint", releases=False)

    assert r.run() is False
    assert len(created) == 1, (
        f"created {len(created)} resources while the first was still billing")
    assert r.ev["abort_reason"] == "unconfirmed_release"
    assert not r.ev["draws"][0]["release"]["confirmed_not_billing"]
    assert "refusing to create a second resource" in " ".join(r.ev["said"])


def test_F_every_abandoned_resource_keeps_its_own_record(monkeypatch):
    """`ev["pod_id"]` describes the CURRENT pod and is rewritten by the next
    create. An abandoned pod that left no record is a cost nobody can
    reconcile."""
    r, created = _runner(monkeypatch, outcome="cold")
    r.run()

    draws = r.ev["draws"]
    assert len(draws) == 2, "two abandoned draws, two records"
    assert [d["draw"] for d in draws] == [1, 2]
    assert len({d["pod_id"] for d in draws}) == 2, "records collapsed onto one id"
    for d in draws:
        assert d["release"]["confirmed_not_billing"] is True
        assert d["release"]["pod_id"] == d["pod_id"]
        assert d["watchdog_journal"] == f"watchdog_{d['pod_id']}.jsonl", (
            "an abandoned resource's watchdog evidence must be identifiable")


def test_G_all_draws_share_one_ceiling(monkeypatch):
    """Three draws do not buy three ceilings. `start_epoch` is set once, so
    `elapsed()` and `usd()` span the session."""
    r, created = _runner(monkeypatch, outcome="cold")
    r.run()

    starts = {d["cumulative_usd_at_release"] for d in r.ev["draws"]}
    assert len(r.ev["draws"]) == 2
    assert all(isinstance(v, float) for v in starts)
    #: The cost recorded at the second release is not lower than at the first:
    #: it accumulates across resources rather than restarting with each pod.
    usd = [d["cumulative_usd_at_release"] for d in r.ev["draws"]]
    assert usd == sorted(usd), f"cost restarted between draws: {usd}"


def test_H_a_single_draw_still_tears_down_without_redrawing(monkeypatch):
    """`--host-draws 1` remains legal and behaves exactly as it did."""
    r, created = _runner(monkeypatch, outcome="cold", host_draws=1)

    assert r.run() is False
    assert len(created) == 1
    assert r.ev.get("teardown") == ["setup cold"]
    assert "redrawing" not in " ".join(r.ev.get("said", []))




def test_I_a_create_failure_calls_create_once_and_returns_for_review(monkeypatch):
    """No id was returned, so no resource exists and nothing is billing."""
    r, created = _runner(monkeypatch, outcome="ok", create_ok=False)

    assert r.run() is False
    assert len(created) == 1, "C1 must not retry provider creation"
    assert not r.ev.get("slept_for_stock"), "C1 must not wait on changing stock"
    assert r.pod_id == ""
    assert not r.provider.terminated, "nothing was created, so nothing to remove"
    assert not r.ev.get("teardown")


def test_J_a_setup_failure_that_is_not_acquisition_ends_the_session(monkeypatch):
    """Draws replace an unusable HOST. A suite that failed, a gate that refused
    or a driver that died are not acquisition problems, and redrawing one would
    be retrying a deterministic failure against a fresh pod."""
    r, created = _runner(monkeypatch, outcome="tests_failed")
    assert r.run() is False
    assert len(created) == 1, "a non-acquisition failure must not redraw"
    assert r.ev.get("teardown") == ["setup tests_failed"]
    assert not r.ev.get("draws"), "nothing was abandoned, so nothing to record"


# --- the generic runner keeps multi-draw for everyone else -------------------

def test_K_the_generic_runner_still_supports_multiple_draws():
    """The acquisition loop is shared. Five other sessions depend on it."""
    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert "for draw in range(1, self.a.host_draws + 1)" in src
    assert "for attempt in range(1, self.a.create_attempts + 1)" in src


@pytest.mark.parametrize("launcher", [
    "autoinit_phase_a_launch", "autoinit_phase_b_launch",
    "autoinit_continuation_launch", "autoinit_preflight_launch",
])
def test_K2_other_launchers_are_untouched(launcher):
    from session_specs import load_session_launcher

    mod = load_session_launcher(launcher)
    argv = ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"]
    if launcher == "autoinit_continuation_launch":
        argv += ["--transport", "relay"]
    args = mod.build_parser().parse_args(argv)
    assert args.host_draws > 1, f"{launcher} lost multi-draw acquisition"
    assert args.create_attempts > 1


def test_K3_every_session_gained_the_confirmed_release():
    """The repair is in the SHARED loop, so it is not a C1 privilege.

    The old branch was unreachable for C1 and reachable for everybody else,
    which means every other session had been redrawing on an unconfirmed
    release. Fixing it in `SessionRunner` fixes it for all of them.

    Read from the AST, not from the source text. A text scan found
    `self.pod_id = ""` inside the comment that EXPLAINS the old defect and
    concluded the defect was still there — a check that cannot tell an
    explanation from the thing it explains.
    """
    import ast

    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    tree = ast.parse(src)
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    #: The redraw branch: the only `if` in `run` that tests the draw index.
    branch = next(n for n in ast.walk(run)
                  if isinstance(n, ast.If) and "host_draws" in ast.unparse(n.test)
                  and "outcome" in ast.unparse(n.test))
    body = [ast.unparse(stmt) for stmt in branch.body]
    confirm = next(i for i, line in enumerate(body)
                   if "release_and_confirm" in line)
    cleared = next(i for i, line in enumerate(body)
                   if line.replace(" ", "") == "self.pod_id=''")
    assert confirm < cleared, (
        f"pod_id is cleared at statement {cleared} and release confirmed at "
        f"{confirm}; clearing first is the defect")
    guard = next(i for i, line in enumerate(body)
                 if "confirmed_not_billing" in line and line.startswith("if"))
    assert confirm < guard < cleared, (
        "the refusal must sit between confirming and clearing")


# --- the redraw branch is reachable, and bounded -----------------------------

def test_L_the_redraw_branch_is_reachable_and_capped():
    """It was unreachable by construction, which hid that it was also wrong."""
    host_draws = C1.build_parser().parse_args(BASE_ARGV).host_draws
    assert host_draws == C1.C1_MAX_HOST_DRAWS == 3
    assert 1 < host_draws, "the cold/no_endpoint redraw branch is unreachable"


def test_M_the_live_grant_and_the_launcher_agree_on_acquisition():
    """Against the LIVE grant, not attempt 9's.

    This read `logs/budget/approvals/autoinit_c1_attempt9_grant.json`, a frozen artifact of a
    session that ran under the superseded rule. A contract test pointed at
    sealed evidence can only ever re-assert history.
    """
    import json

    #: Across the canonical layout: stage-scoped and unscoped alike.
    live = sorted((REPO / "logs/runs").glob("*/*/*/governance/grant.json"))
    assert live, "no grant exists in any run directory"
    grant = json.loads(live[-1].read_text())
    one_use = grant["one_use"]
    assert one_use["issuances_permitted"] == 1
    assert one_use["launch_attempts_permitted"] == 1
    #: Resources per attempt is now a RANGE the launcher caps, not a constant
    #: the grant restates: the invariant that survived is one BILLING resource
    #: at a time, enforced by confirmed release rather than by arithmetic.
    assert one_use["provider_resources_permitted"] <= C1.C1_MAX_HOST_DRAWS
    assert one_use.get("one_billing_resource_at_a_time") is True, (
        "the live grant does not state the invariant the launcher enforces")


# --- one resource, one journal, from the first tick --------------------------
#
# `Journal.write` reopens by PATH on every event. Renaming a live path aside
# therefore isolates nothing: an old watchdog that has not yet exited recreates
# the shared path and writes its final poll and `watchdog_end` into the NEXT
# resource's journal, while its own archive is left without an ending.
#
# Reproduced below with two real processes on a real filesystem, in the order a
# redraw actually produces: old writer still alive, new writer starts, old
# writer finishes.

def _journal_proc(path, pod, events, delay=0.0):
    """A real detached writer using the REAL Journal, like the watchdog does."""
    import subprocess as sp
    import sys as _s

    code = (
        "import sys, time;"
        "sys.path.insert(0, %r);"
        "from aadistill.infrastructure.watchdog import Journal;"
        "j = Journal(%r);"
        "time.sleep(%r);"
        "[j.write(e, pod_id=%r) for e in %r]" % (
            str(REPO / "src"), str(path), delay, pod, events))
    return sp.Popen([_s.executable, "-c", code])


def test_a_slow_previous_watchdog_cannot_write_into_the_next_journal(tmp_path):
    """The defect, at the file level, with the paths the runner now derives."""
    from aadistill.infrastructure.session_runner import SessionRunner

    old_pod, new_pod = "podOLD", "podNEW"
    old_path = tmp_path / SessionRunner._watchdog_journal_name(old_pod)
    new_path = tmp_path / SessionRunner._watchdog_journal_name(new_pod)
    assert old_path != new_path, "two resources share one journal path"

    # The old writer is still alive and finishes AFTER the new one starts.
    old = _journal_proc(old_path, old_pod, ["poll", "watchdog_end"], delay=1.0)
    new = _journal_proc(new_path, new_pod, ["watchdog_start", "poll"], delay=0.0)
    new.wait(timeout=60)
    old.wait(timeout=60)

    old_lines = [json.loads(x) for x in old_path.read_text().splitlines() if x]
    new_lines = [json.loads(x) for x in new_path.read_text().splitlines() if x]

    assert {e["event"] for e in old_lines} == {"poll", "watchdog_end"}, (
        "the abandoned resource's archive is missing its ending")
    assert all(e["pod_id"] == old_pod for e in old_lines)
    assert all(e["pod_id"] == new_pod for e in new_lines), (
        "the previous resource's events landed in the next resource's journal")
    assert "watchdog_end" not in {e["event"] for e in new_lines}


def test_the_runner_derives_a_distinct_journal_per_pod():
    """And not by renaming: the name is the pod's from the first tick."""
    from aadistill.infrastructure.session_runner import SessionRunner

    names = {SessionRunner._watchdog_journal_name(p) for p in ("a", "b", "c")}
    assert len(names) == 3
    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    body = src[src.index("def launch_watchdog"):src.index("def wait_endpoint")]
    assert ".rename(" not in body, (
        "launch_watchdog still renames a path a live writer may hold")
    assert '"watchdog.jsonl"' not in body, "a shared journal path is back"


def test_each_draw_keeps_its_own_raw_provider_response(monkeypatch):
    """`attempt` restarts at 1 inside every draw, so draw 2's first create
    overwrote draw 1's raw response -- the only record of what the provider
    said, including a refusal that returned no pod id at all."""
    r, created = _runner(monkeypatch, outcome="no_endpoint")
    for f in r.scr.glob("create_raw_*"):
        f.unlink()
    r.run()

    raw = sorted(p.name for p in r.scr.glob("create_raw_*"))
    assert len(raw) == len(created), f"{len(created)} creates left {raw}"
    assert len(set(raw)) == len(raw), f"raw responses overwrote each other: {raw}"


def test_a_create_that_returned_no_id_is_still_recorded(monkeypatch):
    """It created nothing and billed nothing, but it happened, and the
    provider's refusal is the only evidence of why."""
    r, created = _runner(monkeypatch, outcome="ok", create_ok=False)

    assert r.run() is False
    assert len(created) == 1
    draws = r.ev.get("draws") or []
    assert len(draws) == 1, "a failed create left no draw record"
    assert draws[0]["outcome"] == "create_failed"
    assert draws[0]["pod_id"] is None
    assert draws[0]["release"]["provider_resource_created"] is False
    #: And it names the files that are really there, so the refusal can be
    #: read from the record without guessing how many attempts were made.
    named = draws[0]["release"]["raw_responses"]
    assert named == ["create_raw_d1_a1.txt"]
    assert all((r.scr / n).is_file() for n in named), named

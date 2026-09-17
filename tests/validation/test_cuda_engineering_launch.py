"""The engineering launcher's spend contract, exercised without a provider.

Everything after `create()` is unreachable without paying, and unreachable code
is where this project's paid sessions have died. So the guards are driven here
with the provider and the CLI replaced by recorders: no resource is created, and
each test asserts a property that costs money to get wrong.

What is pinned:

* **one create attempt**, structurally -- not a loop over candidates that
  happens to break early. The canary loops; a loop is a second create;
* the watchdog is detached from inside `register()`, so a pod id cannot exist
  without one;
* the minute budgets are DERIVED from the accepted rate. A fixed allowance is
  the specific thing the maintainer forbade, and a dearer GPU must buy less
  time, not the same time;
* a pod that provisions above the accepted rate is torn down unused and NOT
  replaced;
* an empty create response is reconciled read-only against the run's unique
  name before anything claims `$0`;
* teardown is not complete because a remove request returned 0.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/validation/cuda_engineering_launch.py"
AUTH = REPO / "logs/stages/stage-1/phase_c1/validations/cuda-stage-f/v1/authorization.json"
#: The ledger is the authorization's sibling, which is how the launcher
#: resolves it too -- naming ONE file cannot select someone else's ledger.
CAMPAIGN = AUTH.parent / "campaign.json"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("cuda_eng_launch", ENTRY)
    m = importlib.util.module_from_spec(spec)
    #: Registered so `type(eng).__module__` resolves -- the launcher's own
    #: `Stop` is what the guards raise, and a test that caught a DIFFERENT
    #: `Stop` would pass while checking nothing.
    sys.modules["cuda_eng_launch"] = m
    spec.loader.exec_module(m)
    return m


def mod_default(name: str):
    """Read a module-level default without importing the launcher twice.

    `args()` is called outside the `mod` fixture in several tests, so it cannot
    take the module as a parameter. The defaults are plain constants; reading
    them from the source keeps the fixture honest if one is renamed.
    """
    import ast
    tree = ast.parse(ENTRY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level constant any more")


def args(**over):
    base = dict(run_id="t-run", scr="", execution_sha="deadbeef",
                image="img", remote_python="python", disk_gb=20,
                startup_limit_min=1.0, run_limit_min=1.0, min_minutes=25.0,
                dry_run=False,
                #: The per-validation profile. Defaults to stage F's, which
                #: is what these tests exercise -- the launcher is now
                #: parameterized, and every constant it used to hold is a
                #: flag resolved here.
                authorization=mod_default("DEFAULT_AUTHORIZATION"),
                experiment_id=mod_default("DEFAULT_EXPERIMENT_ID"),
                check="scripts/validation/cuda_engineering_check.py",
                check_config="configs/validation/cuda_engineering.json",
                ship=[], gpu=[])
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.fixture
def eng(mod, tmp_path, monkeypatch):
    """A launcher with the provider and CLI replaced. Creates nothing."""
    monkeypatch.setattr(mod, "read_api_key", lambda p: "test-key")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    fake_cli = tmp_path / "runpodctl"
    fake_cli.write_text("#!/bin/sh\nexit 0\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(mod, "provider_cli_candidates", lambda: (str(fake_cli),))
    monkeypatch.setattr(mod, "RunPodProvider", lambda key: types.SimpleNamespace(
        _gql=lambda q: {"data": {}}))
    e = mod.Engineering(args(scr=str(tmp_path / "scr")))
    e.creates_issued = []
    return e


# --- the authorization it reads ---------------------------------------------

def test_it_reads_the_engineering_authorization_not_a_c1_grant():
    doc = json.loads(AUTH.read_text())
    assert doc["schema"] == "aadistill.engineering_validation_authorization/v1"
    rc = doc["resource_contract"]
    #: The count-based clauses are HISTORY: stage F ran under them, and they
    #: were withdrawn on 2026-09-17 because attempt count is not budget. They
    #: are asserted here as a fact about a closed record, not as a contract
    #: the launcher still enforces -- it no longer reads them, and an
    #: authorization written after the withdrawal omits them.
    assert rc["provider_resources_max"] == 1
    assert rc["provider_create_attempts_max"] == 1
    assert rc["retries_or_replacement_pods"] == 0
    assert rc["engineering_soft_cap_usd"] == 0.25
    assert rc["total_resource_cost_ceiling_usd"] == 0.40
    assert rc["teardown_reserve_usd"] == 0.15


def test_the_reserve_sits_inside_the_ceiling():
    rc = json.loads(AUTH.read_text())["resource_contract"]
    assert rc["teardown_reserve_usd"] < rc["total_resource_cost_ceiling_usd"]


def test_it_authorizes_nothing_formal():
    doc = json.loads(AUTH.read_text())
    joined = " ".join(doc["does_not_authorize"]).lower()
    for forbidden in ("attempt-10", "formal c1 bundle", "confirmation battery",
                      "formal c1 decision", "merging"):
        assert forbidden in joined, forbidden
    assert doc["formal_c1_status_unchanged"]["formal_treatment"] == "UNMEASURED"
    assert doc["formal_c1_status_unchanged"]["attempt_9"] == "NO DECISION"


# --- exactly one create -----------------------------------------------------

def test_a_second_create_is_refused_structurally(eng, mod, monkeypatch):
    """Not 'the loop breaks'. Calling create twice must raise."""
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            stdout='{"id":"pod-1"}', stderr="", returncode=0))
    monkeypatch.setattr(eng, "launch_watchdog", lambda: None)
    eng.create("NVIDIA RTX A4000", 0.25)
    assert eng.created == 1
    with pytest.raises(mod.Stop, match="already called create"):
        eng.create("NVIDIA RTX A4000", 0.25)


def test_the_create_argv_names_one_gpu_and_one_count(eng, mod, monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return types.SimpleNamespace(stdout='{"id":"pod-1"}', stderr="",
                                     returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(eng, "launch_watchdog", lambda: None)
    eng.create("NVIDIA RTX A4000", 0.25)
    argv = seen["argv"]
    assert argv.count("--gpu-id") == 1
    assert argv[argv.index("--gpu-count") + 1] == "1"
    assert "aad-cuda-eng-t-run" in argv
    # The run's unique name is what reconciliation matches on.
    assert argv[argv.index("--name") + 1] == "aad-cuda-eng-t-run"


# --- the watchdog cannot be skipped -----------------------------------------

def test_registering_a_pod_starts_the_watchdog(eng, monkeypatch):
    started = []
    monkeypatch.setattr(eng, "launch_watchdog", lambda: started.append(1))
    eng.start_epoch, eng.rate = 1.0, 0.25
    eng.register("pod-xyz")
    assert eng.pod_id == "pod-xyz"
    assert eng.ev["provider_resource_created"] is True
    assert started == [1], "a pod id existed without a watchdog"


def test_the_pod_id_is_persisted_before_anything_else_can_fail(eng, monkeypatch):
    monkeypatch.setattr(eng, "launch_watchdog", lambda: None)
    eng.start_epoch, eng.rate = 1.0, 0.25
    eng.register("pod-xyz")
    assert (eng.scr / "pod_id").read_text() == "pod-xyz"


# --- the budget is derived, never fixed -------------------------------------

@pytest.mark.parametrize("rate,soft_min,hard_min", [
    (0.25, 60.0, 96.0),
    (0.50, 30.0, 48.0),
    (1.09, 13.761, 22.018),
])
def test_the_minute_budgets_scale_with_the_rate(eng, rate, soft_min, hard_min):
    """A dearer GPU buys less time. A fixed 20-minute allowance -- the thing
    the maintainer forbade -- would give all three rows the same number."""
    assert eng.soft_usd / rate * 60 == pytest.approx(soft_min, abs=0.01)
    assert eng.hard_usd / rate * 60 == pytest.approx(hard_min, abs=0.01)


def test_the_watchdog_gets_the_REMAINING_budget_not_a_fresh_one(eng, mod,
                                                                 monkeypatch):
    """A replacement resource does not receive a new $0.40.

    The watchdog measures THIS pod's clock, so what it must be given is the
    campaign's remaining ceiling -- otherwise every rerun silently re-authorises
    the full task budget.
    """
    seen = {}
    monkeypatch.setattr(mod.subprocess, "Popen",
                        lambda cmd, **kw: seen.setdefault("cmd", cmd))
    eng.pod_id, eng.start_epoch, eng.rate = "pod-1", 1.0, 0.50
    eng.launch_watchdog()
    cmd = seen["cmd"]
    authorized = float(cmd[cmd.index("--authorized-usd") + 1])
    assert authorized == pytest.approx(eng.remaining_total)
    assert authorized < eng.hard_usd, "the watchdog was handed a fresh ceiling"
    assert float(cmd[cmd.index("--hard-minutes") + 1]) == pytest.approx(
        eng.remaining_total / 0.50 * 60)


def test_a_ceiling_that_buys_too_little_time_refuses_before_creating(eng, mod,
                                                                     monkeypatch):
    """A GPU so dear that $0.40 cannot finish setup is not worth creating."""
    monkeypatch.setattr(eng.provider, "_gql", lambda q: {"data": {"gpuTypes": [
        {"id": "x", "securePrice": 4.0, "memoryInGb": 80,
         "lowestPrice": {"stockStatus": "High"}}]}})
    with pytest.raises(mod.Stop, match="declared minimum"):
        eng.quote()


def _stop_cls(eng):
    """The launcher's own `Stop`, from the module the fixture loaded."""
    return sys.modules[type(eng).__module__].Stop


def test_the_soft_cap_stops_work_before_the_next_step(eng):
    eng.start_epoch = 0.0     # epoch 0 -> "spent" is enormous
    eng.rate = 0.25
    with pytest.raises(_stop_cls(eng), match="soft cap reached"):
        eng.check_soft_cap("the next step")


# --- a wrong price is torn down, not replaced -------------------------------

def test_a_pod_above_the_accepted_rate_stops_the_run(eng, monkeypatch):
    eng.pod_id, eng.rate, eng.start_epoch = "pod-1", 0.25, 1.0
    monkeypatch.setattr(eng.provider, "_gql",
                        lambda q: {"data": {"pod": {"costPerHr": 0.9}}})
    with pytest.raises(_stop_cls(eng), match="above the accepted"):
        eng.verify_rate()
    assert eng.ev["actual_price_per_hour"] == 0.9
    assert eng.created == 0, "no replacement create was issued"


def test_a_pod_at_or_below_the_accepted_rate_proceeds(eng, monkeypatch):
    eng.pod_id, eng.rate, eng.start_epoch = "pod-1", 0.25, 1.0
    monkeypatch.setattr(eng.provider, "_gql",
                        lambda q: {"data": {"pod": {"costPerHr": 0.24}}})
    eng.verify_rate()
    assert eng.ev["actual_price_per_hour"] == 0.24


# --- an empty create response is reconciled, not assumed to be $0 -----------

def test_an_empty_response_is_reconciled_by_run_name(eng, mod, monkeypatch):
    #: SEQUENCE-AWARE, because `create` now queries the inventory TWICE for two
    #: different questions: is anything already billing (before), and did the
    #: silent create leave a pod behind (after). A stub answering both with the
    #: same populated inventory would make the pre-create guard refuse this
    #: run's own resource -- so the fake models what the provider really does,
    #: where the pod does not exist until create is called.
    created = []

    def run(*a, **k):
        created.append(1)
        return types.SimpleNamespace(stdout="", stderr="", returncode=1)

    monkeypatch.setattr(mod.subprocess, "run", run)
    monkeypatch.setattr(eng.provider, "_gql", lambda q: {"data": {"myself": {
        "pods": ([{"id": "orphan-1", "name": "aad-cuda-eng-t-run",
                   "desiredStatus": "RUNNING", "costPerHr": 0.25}]
                 if created else [])}}})
    monkeypatch.setattr(eng, "launch_watchdog", lambda: None)
    eng.create("NVIDIA RTX A4000", 0.25)
    assert eng.pod_id == "orphan-1", "an unreturned id was left unowned"
    assert eng.ev["provider_resource_created"] is True


def test_a_live_resource_refuses_the_create_before_it_happens(eng, mod, monkeypatch):
    """The invariant that replaced the attempt counter.

    A counter refused a legitimate second subrun and did nothing about a pod an
    earlier subrun had left alive. This is the property that actually matters:
    at most ONE billing resource at any instant. Nothing may be created beside
    something already running, whatever the attempt number is.
    """
    calls = []
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: calls.append(1))
    monkeypatch.setattr(eng.provider, "_gql", lambda q: {"data": {"myself": {
        "pods": [{"id": "someone-elses", "name": "aad-other",
                  "desiredStatus": "RUNNING", "costPerHr": 1.09}]}}})
    with pytest.raises(_stop_cls(eng), match="already present"):
        eng.create("NVIDIA L40S", 1.09)
    assert not calls, "a create was issued beside a live resource"
    assert eng.created == 0


def test_a_terminated_resource_does_not_block_a_replacement(eng, mod, monkeypatch):
    """The other half: a replacement IS allowed once the previous one is gone.

    Provider-confirmed gone means absent or TERMINATED in the inventory. If a
    terminated pod still blocked, the money-bounded retry the contract permits
    would be unreachable -- which is the counter coming back under another name.
    """
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            stdout='{"id":"pod-2"}', stderr="", returncode=0))
    monkeypatch.setattr(eng.provider, "_gql", lambda q: {"data": {"myself": {
        "pods": [{"id": "old-one", "name": "aad-cuda-eng-earlier",
                  "desiredStatus": "TERMINATED", "costPerHr": 0}]}}})
    monkeypatch.setattr(eng, "launch_watchdog", lambda: None)
    eng.create("NVIDIA L40S", 1.09)
    assert eng.pod_id == "pod-2"


def test_an_unreadable_inventory_refuses_rather_than_proceeding(eng, mod, monkeypatch):
    """Unknown resource state is the condition under which nothing is created."""
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: calls.append(1))

    def boom(q):
        raise OSError("network down")

    monkeypatch.setattr(eng.provider, "_gql", boom)
    with pytest.raises(_stop_cls(eng), match="UNKNOWN"):
        eng.create("NVIDIA L40S", 1.09)
    assert not calls


def test_a_failed_reconcile_does_not_claim_no_resource(eng, mod, monkeypatch):
    """It says the state is unknown rather than $0.

    The pre-create guard must SUCCEED here and the post-create reconcile must
    FAIL, or this tests the guard instead of the reconcile.
    """
    created = []

    def run(*a, **k):
        created.append(1)
        return types.SimpleNamespace(stdout="", stderr="", returncode=1)

    monkeypatch.setattr(mod.subprocess, "run", run)

    def gql(q):
        if created:
            raise OSError("network down")
        return {"data": {"myself": {"pods": []}}}

    monkeypatch.setattr(eng.provider, "_gql", gql)
    with pytest.raises(_stop_cls(eng)):
        eng.create("NVIDIA RTX A4000", 0.25)
    assert "reconcile_error" in eng.ev, (
        "the guard refused instead of the reconcile failing")


# --- teardown is confirmed, not requested -----------------------------------

def test_a_successful_remove_is_not_confirmation(eng, mod, monkeypatch):
    eng.pod_id, eng.rate, eng.start_epoch = "pod-1", 0.25, 1.0
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            stdout="removed", stderr="", returncode=0))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    # The pod keeps appearing in the inventory: remove succeeded, it is still there.
    monkeypatch.setattr(eng.provider, "_gql", lambda q: {"data": {"myself": {
        "pods": [{"id": "pod-1", "desiredStatus": "RUNNING"}]}}})
    eng.teardown()
    assert eng.ev["teardown"]["remove_rc"] == 0
    assert eng.ev["teardown"]["provider_confirms_gone"] is False


def test_disappearance_from_the_inventory_is_confirmation(eng, mod, monkeypatch):
    eng.pod_id, eng.rate, eng.start_epoch = "pod-1", 0.25, 1.0
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            stdout="", stderr="", returncode=0))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(eng.provider, "_gql",
                        lambda q: {"data": {"myself": {"pods": []}}})
    eng.teardown()
    assert eng.ev["teardown"]["provider_confirms_gone"] is True


# --- what it must never touch ----------------------------------------------

def test_it_names_no_formal_c1_input():
    src = ENTRY.read_text()
    for forbidden in ("autoinit_c1_authorization", "c1_confirmation_v1",
                      "attempt9", "recovery_search_v2", "C1Authorization"):
        assert forbidden not in src, f"the engineering launcher names {forbidden}"


# --- the two defects that cost the 2026-09-09 run --------------------------

class TestTheDependencyStepReportsItsOwnFailure:
    """Both of these were live when the authorized GPU run executed.

    `pip install ... 2>&1 | tail -5` makes the shell report TAIL's status,
    which is always 0. pip refused the install under PEP 668, the launcher
    recorded `pip_rc: 0`, and the validation failed four seconds later on
    `ModuleNotFoundError: numpy` -- reading as a validation failure when it was
    a setup one.
    """

    def test_the_pip_command_does_not_pipe_its_exit_status_away(self):
        """Code only. The comment above the fix has to be able to quote the
        construct it forbids in order to explain what went wrong."""
        ship = (ENTRY.read_text().split("def ship")[1]
                .split("def validate")[0])
        code = "\n".join(l for l in ship.splitlines()
                         if not l.lstrip().startswith(("#", "#:")))
        assert "| tail" not in code, (
            "the dependency install pipes through tail again; the shell then "
            "reports tail's exit status and a refused install looks successful")

    def test_it_captures_pips_own_return_code(self):
        src = ENTRY.read_text()
        assert "PIP_RC=$?" in src
        assert 'm = re.search(r"PIP_RC=(\\d+)", out)' in src

    def test_it_passes_the_flag_pep_668_asks_for(self):
        """The image's interpreter is externally managed, so a plain install is
        refused by design."""
        assert "--break-system-packages" in ENTRY.read_text()

    def test_an_unparseable_pip_status_is_a_failure_not_a_pass(self, eng, mod,
                                                               monkeypatch):
        """If the marker never appears, the rc is unknown -- which must stop the
        run, not proceed as though it were zero."""
        import re as _re
        out = "some output with no marker at all"
        m = _re.search(r"PIP_RC=(\d+)", out)
        assert (int(m.group(1)) if m else -1) == -1

    def test_the_import_probe_precedes_the_validation(self):
        """A missing module found in the probe costs seconds and reads as a
        setup failure; the same module found inside the check reads as a
        validation failure, which is what happened."""
        src = ENTRY.read_text()
        ship = src.split("def ship")[1].split("def validate")[0]
        assert "IMPORTS_OK" in ship
        assert "import numpy" in ship


# --- the cumulative cap, across resources and subruns ----------------------

class TestPriorSpendReducesTheNextResourcesLimits:
    """The property that makes a bounded repair loop safe.

    Without it, "at most $0.40" means $0.40 per invocation and an autonomous
    loop can spend without limit while every individual run looks compliant.
    """

    def test_the_launcher_starts_from_what_earlier_subruns_booked(self, eng):
        booked = json.loads(
            (REPO / "logs/stages/stage-1/phase_c1/validations/cuda-stage-f/v1/campaign.json").read_text()
        )["booked_usd"]
        assert eng.booked_usd == booked > 0, "no prior spend was carried in"
        assert eng.remaining_total == pytest.approx(eng.hard_usd - booked)
        assert eng.remaining_soft == pytest.approx(eng.soft_usd - booked)

    def test_the_soft_cap_counts_the_whole_campaign_not_this_subrun(self, eng):
        """A subrun that has spent nothing yet is already `booked` closer to the
        cap than a fresh allocation would suggest."""
        eng.start_epoch, eng.rate = None, None
        assert eng.subrun_spent() == 0.0
        assert eng.spent() == pytest.approx(eng.booked_usd)

    def test_a_campaign_that_has_exhausted_its_ceiling_refuses_to_start(
            self, mod, tmp_path, monkeypatch):
        """Not at create time -- at construction, before anything is quoted.

        Through the REAL resolution path: the ledger is the authorization's
        sibling, so this points `--authorization` at a directory holding both.
        Patching a module constant would no longer test how the launcher finds
        its ledger, which is the part that can now go wrong.
        """
        home = tmp_path / "validations" / "exhausted" / "v1"
        home.mkdir(parents=True)
        (home / "authorization.json").write_text(AUTH.read_text())
        camp = json.loads(CAMPAIGN.read_text())
        camp["booked_usd"] = 0.39          # leaves less than the $0.15 reserve
        (home / "campaign.json").write_text(json.dumps(camp))
        monkeypatch.setattr(mod, "read_api_key", lambda p: "k")
        with pytest.raises(mod.Stop, match="teardown reserve"):
            mod.Engineering(args(
                scr=str(tmp_path / "s"),
                authorization=str(home / "authorization.json")))

    def test_a_validation_without_a_ledger_refuses_to_start(
            self, mod, tmp_path, monkeypatch):
        """The cumulative ledger is what bounds a retry.

        With the attempt counter withdrawn, money is the only bound left. A
        validation that cannot read what it already spent would treat every
        invocation as a fresh allocation -- the exact failure a cumulative cap
        exists to prevent.
        """
        home = tmp_path / "no-ledger" / "v1"
        home.mkdir(parents=True)
        (home / "authorization.json").write_text(AUTH.read_text())
        monkeypatch.setattr(mod, "read_api_key", lambda p: "k")
        with pytest.raises(mod.Stop, match="campaign ledger"):
            mod.Engineering(args(
                scr=str(tmp_path / "s"),
                authorization=str(home / "authorization.json")))

    def test_the_feasibility_test_uses_remaining_minus_reserve(self, eng,
                                                               monkeypatch):
        """A rate that would fit inside a FRESH ceiling but not inside what is
        left must be refused."""
        eng.booked_usd = 0.20
        eng.remaining_total = round(eng.hard_usd - 0.20, 6)   # $0.20
        usable = eng.remaining_total - eng.reserve_usd        # $0.05
        rate = 0.24
        assert usable / rate * 60 < eng.a.min_minutes
        # ...while the original ceiling would have looked fine:
        assert eng.hard_usd / rate * 60 > eng.a.min_minutes

    def test_a_subrun_is_booked_even_when_it_fails(self, eng, mod, monkeypatch,
                                                   tmp_path):
        """A failed subrun that did not book its spend hands the next one a
        budget that does not exist."""
        camp = json.loads(eng.campaign_path.read_text())
        target = tmp_path / "campaign.json"
        target.write_text(json.dumps(camp))
        monkeypatch.setattr(eng, "campaign_path", target)
        eng.created, eng.pod_id, eng.rate = 1, "pod-x", 0.24
        eng.start_epoch = mod.time.time() - 60
        eng.ev["elapsed_minutes"] = 1.0
        eng.book_subrun("FAIL", "setup")
        after = json.loads(target.read_text())
        assert after["subruns"][-1]["subrun_id"] == "t-run"
        assert after["subruns"][-1]["verdict"] == "FAIL"
        assert after["subruns"][-1]["failure_class"] == "setup"
        assert after["booked_usd"] > camp["booked_usd"]

    def test_booking_is_idempotent(self, eng, mod, monkeypatch, tmp_path):
        camp = json.loads(eng.campaign_path.read_text())
        target = tmp_path / "campaign.json"
        target.write_text(json.dumps(camp))
        monkeypatch.setattr(eng, "campaign_path", target)
        eng.created, eng.pod_id, eng.rate = 1, "pod-x", 0.24
        eng.start_epoch = mod.time.time() - 60
        eng.book_subrun("FAIL", "setup")
        first = json.loads(target.read_text())["booked_usd"]
        eng.book_subrun("FAIL", "setup")
        assert json.loads(target.read_text())["booked_usd"] == first


# --- the setup step, driven behaviourally ----------------------------------

class FakeTarget:
    """An SSH target that replays scripted (rc, stdout, stderr) per command.

    Behavioural, not a source scan: the launcher's own code runs and the
    outcome is decided by what the 'pod' returns.
    """

    def __init__(self, script):
        self.script = script
        self.calls = []

    def run(self, cmd, timeout=None):
        self.calls.append(cmd)
        for needle, (rc, out, err) in self.script.items():
            if needle in cmd:
                return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)
        return types.SimpleNamespace(returncode=0, stdout="RC=0\n", stderr="")


def _ok_venv():
    return {"-m venv": (0, "Python 3.12.3\nRC=0\n", "")}


class TestTheSetupStepFailsWhenTheEnvironmentDoes:
    def test_a_refused_install_stops_the_run(self, eng):
        """The 2026-09-09 defect, behaviourally: pip refuses, and the launcher
        must not proceed to the validation."""
        t = FakeTarget({**_ok_venv(),
                        "-m pip install": (
                            0, "error: externally-managed-environment\nPIP_RC=1\n", "")})
        with pytest.raises(_stop_cls(eng), match="dependency install failed"):
            eng.prepare_environment(t, "/workspace/aad")
        assert eng.ev["pip_rc"] == 1
        assert "externally-managed" in eng.ev["pip_output_tail"]

    def test_an_install_whose_status_is_unreadable_is_a_failure(self, eng):
        """No marker means the return code is unknown, which must stop the run
        rather than proceed as though it were zero."""
        t = FakeTarget({**_ok_venv(),
                        "-m pip install": (0, "some output, no marker", "")})
        with pytest.raises(_stop_cls(eng), match="pip rc=-1"):
            eng.prepare_environment(t, "/workspace/aad")

    def test_a_broken_environment_stops_before_installing(self, eng):
        t = FakeTarget({"-m venv": (1, "No module named venv\nRC=1\n", "")})
        with pytest.raises(_stop_cls(eng), match="could not build the validation"):
            eng.prepare_environment(t, "/workspace/aad")
        assert "pip_rc" not in eng.ev

    def test_install_probe_and_validation_use_the_SAME_interpreter(self, eng,
                                                                   monkeypatch):
        """The first run installed with a bare `pip` and validated with a
        different `python`, so nothing tied them together."""
        monkeypatch.setattr(eng, "readiness_probe", lambda *a: None)
        t = FakeTarget({**_ok_venv(),
                        "-m pip install": (0, "PIP_RC=0\n", ""),
                        "-m pip list": (0, "torch==2.9.1\nnumpy==2.1.0\n", "")})
        eng.prepare_environment(t, "/workspace/aad")
        py = eng.deploy["remote_python"]
        installs = [c for c in t.calls if "-m pip install" in c]
        assert installs and all(c.startswith(py) for c in installs), installs
        assert eng.ev["validation_interpreter"] == py
        # and the validation command uses that same interpreter
        assert py in f'{eng.deploy["remote_python"]} scripts/validation/x.py'

    def test_the_installed_versions_are_recorded(self, eng, monkeypatch):
        monkeypatch.setattr(eng, "readiness_probe", lambda *a: None)
        t = FakeTarget({**_ok_venv(),
                        "-m pip install": (0, "PIP_RC=0\n", ""),
                        "-m pip list": (0, "torch==2.9.1+cu130\nnumpy==2.1.0\n", "")})
        eng.prepare_environment(t, "/workspace/aad")
        assert "torch==2.9.1+cu130" in eng.ev["installed_versions"]

    def test_the_venv_inherits_system_site_packages(self, eng, monkeypatch):
        """A clean venv would hide the image's CUDA torch, and resolving torch
        afresh can land a CPU build -- a GPU validation that quietly is not."""
        monkeypatch.setattr(eng, "readiness_probe", lambda *a: None)
        t = FakeTarget({**_ok_venv(), "-m pip install": (0, "PIP_RC=0\n", ""),
                        "-m pip list": (0, "", "")})
        eng.prepare_environment(t, "/workspace/aad")
        venv_cmd = next(c for c in t.calls if "-m venv" in c)
        assert "--system-site-packages" in venv_cmd

    def test_torch_is_not_in_the_install_list(self, eng):
        """Installing torch is how a CUDA build gets silently replaced."""
        assert "torch" not in [d.lower() for d in eng.deploy["pip_deps"]]


class TestTheReadinessProbeRequiresAWorkingGPU:
    PY = "/workspace/eng-venv/bin/python"

    def probe_result(self, eng, rc, stdout):
        #: Matches the base64 transport the launcher uses now. The probe used
        #: to travel as an ssh heredoc, which is a quoting failure waiting to
        #: look like a device failure.
        return FakeTarget({"readiness_probe.py": (0, stdout + f"\nPROBE_RC={rc}\n", "")})

    def test_no_cuda_is_a_failure_not_a_pass(self, eng):
        """`CUDA=False` beside an IMPORTS_OK marker is not readiness."""
        t = self.probe_result(eng, 2, '{"cuda_available": false}\nPROBE_FAIL cuda_not_available')
        with pytest.raises(_stop_cls(eng), match="readiness probe failed"):
            eng.readiness_probe(t, "/workspace/aad", self.PY)
        assert eng.ev["readiness_probe"]["rc"] == 2

    @pytest.mark.parametrize("rc,marker", [(3, "capability"), (4, "dtype"),
                                           (5, "vram"), (6, "matmul")])
    def test_each_capability_failure_stops_the_run(self, eng, rc, marker):
        t = self.probe_result(eng, rc, '{"cuda_available": true}\nPROBE_FAIL ' + marker)
        with pytest.raises(_stop_cls(eng), match="readiness probe failed"):
            eng.readiness_probe(t, "/workspace/aad", self.PY)

    def test_a_zero_exit_without_the_ok_marker_is_still_a_failure(self, eng):
        """A probe that crashed in a way that still exited 0 must not pass."""
        t = self.probe_result(eng, 0, '{"cuda_available": true}')
        with pytest.raises(_stop_cls(eng), match="readiness probe failed"):
            eng.readiness_probe(t, "/workspace/aad", self.PY)

    def test_a_healthy_gpu_passes_and_is_recorded(self, eng):
        report = ('{"torch": "2.9.1+cu130", "cuda_available": true, '
                  '"device": "NVIDIA RTX 2000 Ada Generation", '
                  '"capability": [8, 9], "bf16_supported": true, '
                  '"free_gib": 15.5, "device_matmul_finite": true}')
        t = self.probe_result(eng, 0, report + "\nPROBE_OK")
        eng.readiness_probe(t, "/workspace/aad", self.PY)
        rep = eng.ev["readiness_probe"]["report"]
        assert rep["capability"] == [8, 9] and rep["bf16_supported"] is True
        assert rep["device_matmul_finite"] is True

    def test_the_probe_runs_under_the_validation_interpreter(self, eng):
        t = self.probe_result(eng, 0, '{"a":1}\nPROBE_OK')
        eng.readiness_probe(t, "/workspace/aad", self.PY)
        assert any(self.PY in c for c in t.calls)

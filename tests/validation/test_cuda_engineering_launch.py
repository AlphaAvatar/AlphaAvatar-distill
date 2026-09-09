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
AUTH = REPO / "logs/validations/cuda-stage-f/v1/authorization.json"

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


def args(**over):
    base = dict(run_id="t-run", scr="", execution_sha="deadbeef",
                image="img", remote_python="python", disk_gb=20,
                startup_limit_min=1.0, run_limit_min=1.0, min_minutes=25.0,
                dry_run=False)
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
    with pytest.raises(mod.Stop, match="no second"):
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


def test_the_watchdog_gets_the_derived_hard_limit(eng, mod, monkeypatch):
    seen = {}
    monkeypatch.setattr(mod.subprocess, "Popen",
                        lambda cmd, **kw: seen.setdefault("cmd", cmd))
    eng.pod_id, eng.start_epoch, eng.rate = "pod-1", 1.0, 0.50
    eng.launch_watchdog()
    cmd = seen["cmd"]
    assert cmd[cmd.index("--authorized-usd") + 1] == "0.4"
    assert float(cmd[cmd.index("--hard-minutes") + 1]) == pytest.approx(48.0)


def test_a_ceiling_that_buys_too_little_time_refuses_before_creating(eng, mod,
                                                                     monkeypatch):
    """A GPU so dear that $0.40 cannot finish setup is not worth creating."""
    monkeypatch.setattr(eng.provider, "_gql", lambda q: {"data": {"gpuTypes": [
        {"id": "x", "securePrice": 4.0, "memoryInGb": 80,
         "lowestPrice": {"stockStatus": "High"}}]}})
    with pytest.raises(mod.Stop, match="buys only"):
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
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            stdout="", stderr="", returncode=1))
    monkeypatch.setattr(eng.provider, "_gql", lambda q: {"data": {"myself": {
        "pods": [{"id": "orphan-1", "name": "aad-cuda-eng-t-run",
                  "desiredStatus": "RUNNING", "costPerHr": 0.25}]}}})
    monkeypatch.setattr(eng, "launch_watchdog", lambda: None)
    eng.create("NVIDIA RTX A4000", 0.25)
    assert eng.pod_id == "orphan-1", "an unreturned id was left unowned"
    assert eng.ev["provider_resource_created"] is True


def test_a_failed_reconcile_does_not_claim_no_resource(eng, mod, monkeypatch):
    """It says the state is unknown rather than $0."""
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            stdout="", stderr="", returncode=1))

    def boom(q):
        raise OSError("network down")

    monkeypatch.setattr(eng.provider, "_gql", boom)
    with pytest.raises(_stop_cls(eng)):
        eng.create("NVIDIA RTX A4000", 0.25)
    assert "reconcile_error" in eng.ev


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

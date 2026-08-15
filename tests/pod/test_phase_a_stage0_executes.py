"""Execute the REAL `PhaseADriver.stage0()` body locally, at $0, on CPU.

Three paid pods died in this stage — `SESSION_KIND` ($0.1075), an engine-probe
argv missing `--model` ($0.4665), and `declared_generation_protocol()` called
with an argument ($0.2103) — for $0.7843 and zero stages passed. All three
survived every existing check for one reason: `test_phase_a_rehearsal.py` drives
the driver's lifecycle with all six stages **scripted**, so the real stage-0 body
had never executed anywhere.

This file executes it. Not a scripted stand-in: the real method, the real
authorization, the real frozen plan, the real `assert_preregistered`, the real
thresholds, the real generation-protocol construction, the real
`RecoveryEvaluationProtocol`, the real Stage-3 binding, and the real attestation
write.

Two substitutions, both boundaries rather than logic:

* the **interpreter path** — `/opt/train/bin/python` exists on the pod, not here.
  The same substitution the continuation rehearsal makes.
* the **vLLM engine probe** — there is no GPU here. It is replaced by the engine
  probe that the Stage-3 session actually recorded, which is the honest stand-in:
  it is what a correctly-configured pod observes.

Scope is stage 0 only. Stages 1-5 remain scripted in the lifecycle rehearsal.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

#: The protocol the Stage-3 controls materialized this run's thresholds under.
STAGE3_HASH = "250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4"
STAGE3_PROBE = REPO / "logs/autoinit_stage3_complete/engine_probe.json"
AUTH = REPO / "logs/autoinit_phase_a_authorization.json"

pytestmark = pytest.mark.skipif(
    not (AUTH.is_file() and STAGE3_PROBE.is_file()
         and (REPO / "logs/autoinit_phase_a_recovery_plan_frozen.json").is_file()
         and (REPO / "artifacts/stage3/recovery_search_v2/manifest.json").is_file()),
    reason="needs the issued authorization, the frozen plan, the Stage-3 engine "
           "probe and the staged battery")


def load_driver(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "phase_a_driver_s0", REPO / "scripts/pod/autoinit_phase_a_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_a_driver_s0"] = mod
    spec.loader.exec_module(mod)
    mod.WS = tmp_path
    mod.STATUS = tmp_path / "phase_a.status"
    mod.AUDIT = tmp_path / "audit"
    (mod.AUDIT / "probes").mkdir(parents=True, exist_ok=True)
    return mod


class Args:
    stage = "all"
    image_digest = "sha256:rehearsal"
    rate = 0.99
    spent_usd = 0.20
    soft_stop_usd = 19.68
    authorized_usd = 20.02
    search_minutes = 180.0
    probe_train_minutes = 61.55
    probe_battery_minutes = 9.82


def build(tmp_path, *, probe_override=None, probe_missing=None):
    """The real driver, with only the two boundaries substituted."""
    mod = load_driver(tmp_path)
    d = mod.PhaseADriver.__new__(mod.PhaseADriver)
    d.a = Args()
    d.t0 = __import__("time").time()
    d.results, d.ev = {}, {"stages": {}}
    d.evaluation_protocol = None
    d.plan = None
    d.leaves, d.control_state, d.rung1, d.rung2 = [], None, None, None
    d.auth = mod.PhaseAAuthorization.load(AUTH)

    real_gate = mod.PhaseADriver.gate.__get__(d)

    def gate(name, argv, *, timeout, python="/opt/train/bin/python"):
        if name == "engine_probe":
            # The only stubbed boundary: no GPU here. Use what Stage 3 observed.
            observed = json.loads(STAGE3_PROBE.read_text())
            if probe_override:
                observed.update(probe_override)
            for key in (probe_missing or ()):
                observed.pop(key, None)
            (mod.AUDIT / "engine_probe.json").write_text(json.dumps(observed))
            return subprocess.CompletedProcess(argv, 0, "", "")
        # Everything else runs for real; only the interpreter path is swapped,
        # exactly as the continuation rehearsal does.
        return real_gate(name, argv, timeout=timeout, python=sys.executable)

    d.gate = gate
    for n in ("enter", "record", "usd", "afford", "save", "child_env"):
        setattr(d, n, getattr(mod.PhaseADriver, n).__get__(d))
    return d, mod


# --- the happy path ---------------------------------------------------------


def test_the_real_stage0_body_runs_end_to_end_and_writes_the_attestation(tmp_path):
    d, mod = build(tmp_path)
    ok = mod.PhaseADriver.stage0(d)
    assert ok is True, d.ev["stages"].get("0", {}).get("reason")

    # It reached the end: the attestation exists and is bound both ways.
    attested = json.loads((mod.AUDIT / "attested_evaluation_protocol.json").read_text())
    assert attested["evaluation_protocol_hash"] == STAGE3_HASH
    assert attested["stage3_evaluation_protocol_hash"] == STAGE3_HASH
    assert attested["bound_to_stage3_thresholds"] is True

    # The frozen science plan was bound, and the thresholds are the Stage-3 ones.
    assert attested["science_plan_hash"] == (
        "02be33b9a7a8e26bc8bfb75795351e8cdc9ffd441b47066cc81887cfc511b55c")
    assert attested["equivalence_interval"] == pytest.approx(0.011695296982299022)
    assert attested["feasibility_floor"] == pytest.approx(0.30)

    # And the live object the later stages use is the same protocol.
    assert d.evaluation_protocol.evaluation_protocol_hash == STAGE3_HASH
    assert d.plan.plan_hash == attested["science_plan_hash"]


def test_the_generation_protocol_is_fully_materialized(tmp_path):
    """`require_materialized` is what attempt 3 never reached."""
    d, mod = build(tmp_path)
    assert mod.PhaseADriver.stage0(d) is True
    # Raises if any declared field is still unset.
    d.evaluation_protocol.generation.require_materialized(context="test")


# --- the Stage-3 binding, which is the scientific gate ----------------------


def test_a_protocol_that_is_not_the_stage3_one_is_refused(tmp_path):
    """The thresholds came from the Stage-3 controls. A candidate measured under
    any other protocol is being judged against numbers that do not describe it,
    so mutual consistency with this session's own attestation is not enough."""
    d, mod = build(tmp_path, probe_override={"vllm_version": "0.0.0-not-stage3"})
    assert mod.PhaseADriver.stage0(d) is False
    reason = d.ev["stages"]["0"]["reason"]
    assert "does not match the Stage-3 protocol" in reason
    assert STAGE3_HASH in reason
    assert not (mod.AUDIT / "attested_evaluation_protocol.json").is_file(), (
        "a refused stage 0 must not leave an attestation behind")


@pytest.mark.parametrize("field", ["tokenizer_sha256", "resolved_context",
                                   "chat_template_sha256"])
def test_any_changed_engine_observation_breaks_the_binding(tmp_path, field):
    """Every observed field enters the protocol hash; none is cosmetic."""
    d, mod = build(tmp_path, probe_override={field: "changed"})
    assert mod.PhaseADriver.stage0(d) is False
    assert "Stage-3 protocol" in d.ev["stages"]["0"]["reason"]


def test_an_unmaterialized_protocol_is_refused(tmp_path):
    """`require_materialized` must be what catches a null observation.

    A *missing key* is the wrong probe for this: it raises `KeyError` inside the
    argument construction, so the stage fails whether or not
    `require_materialized` is called at all — verified by mutation, where
    deleting that call left this test green. `materialized()` accepts `None`
    without complaint, so a **null** field is the case only the explicit check
    can catch.
    """
    d, mod = build(tmp_path, probe_override={"runtime_digest": None})
    assert mod.PhaseADriver.stage0(d) is False
    reason = d.ev["stages"]["0"]["reason"]
    assert "not materialized" in reason, reason
    assert "runtime_digest" in reason


def test_a_missing_engine_observation_also_fails_closed(tmp_path):
    """The other shape: a key the probe never wrote at all."""
    d, mod = build(tmp_path, probe_missing=["stop_token_ids"])
    assert mod.PhaseADriver.stage0(d) is False
    assert "generation protocol" in d.ev["stages"]["0"]["reason"]


def test_a_swapped_stage3_thresholds_artifact_is_refused(tmp_path, monkeypatch):
    """The binding reads the hash from the artifact the thresholds came from, so
    the artifact itself is pinned too — otherwise swapping it would move the
    target rather than fail."""
    d, mod = build(tmp_path)
    doctored = tmp_path / "doctored_thresholds.json"
    payload = json.loads(mod.STAGE3_THRESHOLDS.read_text())
    payload["evaluation_protocol_hash"] = "0" * 64
    doctored.write_text(json.dumps(payload))
    monkeypatch.setattr(mod, "STAGE3_THRESHOLDS", doctored)
    assert mod.PhaseADriver.stage0(d) is False
    assert "not the pinned" in d.ev["stages"]["0"]["reason"]


def test_the_pinned_hash_is_the_one_the_thresholds_were_materialized_under():
    """Guards the constant against drifting away from the artifact."""
    recorded = json.loads(
        (REPO / "logs/autoinit_stage3_complete/materialized_thresholds.json")
        .read_text())["evaluation_protocol_hash"]
    mod_src = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    assert recorded == STAGE3_HASH
    assert STAGE3_HASH in mod_src


# --- the engine-probe argv, which killed attempt 2 --------------------------


def test_stage0_builds_the_engine_probe_argv_with_every_required_flag(tmp_path):
    """Attempt 2 died on a missing --model. Capture the real argv this time."""
    seen = {}
    d, mod = build(tmp_path)
    inner = d.gate

    def capture(name, argv, *, timeout, python="/opt/train/bin/python"):
        if name == "engine_probe":
            seen["argv"], seen["python"] = list(argv), python
        return inner(name, argv, timeout=timeout, python=python)

    d.gate = capture
    assert mod.PhaseADriver.stage0(d) is True
    argv = seen["argv"]
    for flag in ("--model", "--out", "--image-digest"):
        assert flag in argv, f"the engine probe argv omits {flag}"
    assert seen["python"] == "/opt/vllm/bin/python", (
        "the probe must run in the vLLM environment, not the train one")

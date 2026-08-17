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


def build(tmp_path, *, probe_override=None, probe_missing=None,
          runtime_override=None):
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
            if runtime_override:
                observed["runtime"] = {**observed["runtime"], **runtime_override}
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
    assert "not comparable" in reason
    assert "generation_runtime_comparability@v2" in reason
    assert STAGE3_HASH in reason
    assert not (mod.AUDIT / "attested_evaluation_protocol.json").is_file(), (
        "a refused stage 0 must not leave an attestation behind")


@pytest.mark.parametrize("field", ["tokenizer_sha256", "resolved_context",
                                   "chat_template_sha256"])
def test_any_changed_engine_observation_breaks_the_binding(tmp_path, field):
    """Every observed field enters the protocol hash; none is cosmetic."""
    d, mod = build(tmp_path, probe_override={field: "changed"})
    assert mod.PhaseADriver.stage0(d) is False
    assert "not comparable" in d.ev["stages"]["0"]["reason"]


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


# --- generation_runtime_comparability@v2 ------------------------------------
#
# The rule that closed attempt 4: a driver PATCH within a branch is provenance,
# a driver BRANCH change is a real runtime event, and every generation-semantic
# field stays material.

ATTEMPT4_IMAGE = "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.126.09"
STAGE3_IMAGE = "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.159.03"


def test_the_attempt4_driver_patch_now_passes(tmp_path):
    """The exact host that refused attempt 4 at $0.2052 must now be comparable.

    Same image tag, same user-space stack, same engine semantics -- only the
    NVIDIA driver patch differs, within the 580 branch.
    """
    d, mod = build(tmp_path, runtime_override={"image_digest": ATTEMPT4_IMAGE})
    assert mod.PhaseADriver.stage0(d) is True, d.ev["stages"]["0"].get("reason")
    att = json.loads((mod.AUDIT / "attested_evaluation_protocol.json").read_text())
    c = att["comparability"]
    assert c["identities_equal"] is True
    assert c["driver_branch_equal"] is True
    assert c["driver_patch_differs"] is True
    assert c["live_driver"] == "580.126.09"
    assert c["historical_driver"] == "580.159.03"
    # The thresholds still bind to the UNTOUCHED historical protocol.
    assert att["stage3_evaluation_protocol_hash"] == STAGE3_HASH


def test_a_driver_branch_change_still_fails_closed(tmp_path):
    """Not a claim that drivers are universally irrelevant."""
    d, mod = build(tmp_path, runtime_override={
        "image_digest": "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@600.10.01"})
    assert mod.PhaseADriver.stage0(d) is False
    reason = d.ev["stages"]["0"]["reason"]
    assert "driver branch moved" in reason and "580" in reason and "600" in reason


def test_a_different_image_ref_still_fails_closed(tmp_path):
    """The image ref is material even when the digest form is the fallback."""
    d, mod = build(tmp_path, runtime_override={
        "image_digest": "runpod/pytorch:9.9.9-someother-image@580.159.03"})
    assert mod.PhaseADriver.stage0(d) is False
    assert "not comparable" in d.ev["stages"]["0"]["reason"]


@pytest.mark.parametrize("field,value", [
    ("torch_version", "2.99.0+cu130"),
    ("transformers_version", "9.9.9"),
    ("python_version", "3.13.0"),
    ("cuda_runtime", "12.1"),
    ("attention_backend", "flash"),
])
def test_a_user_space_stack_change_still_fails_closed(tmp_path, field, value):
    d, mod = build(tmp_path, runtime_override={field: value})
    assert mod.PhaseADriver.stage0(d) is False
    assert "not comparable" in d.ev["stages"]["0"]["reason"]


@pytest.mark.parametrize("field,value", [
    ("dtype", "float16"),
    ("gpu_memory_utilization", 0.5),
    ("max_num_seqs", 64),
    ("max_num_batched_tokens", 4096),
    ("enforce_eager", True),
    ("resolved_context", 4096),
    ("context_source", "architectural"),
])
def test_an_engine_semantics_change_still_fails_closed(tmp_path, field, value):
    """Fails closed by one of TWO gates, and either is correct.

    Fields the protocol *declares* (dtype, gpu_memory_utilization) are refused
    earlier still, by `materialized()`, as generation-protocol drift -- an
    observation contradicting a declaration. Fields that are purely observed
    reach the v2 comparability check and are refused there. What must never
    happen is that an engine-semantics change passes.
    """
    d, mod = build(tmp_path, probe_override={field: value})
    assert mod.PhaseADriver.stage0(d) is False
    reason = d.ev["stages"]["0"]["reason"]
    assert ("not comparable" in reason
            or "generation-protocol drift" in reason), reason


def test_the_compat_artifact_must_bind_to_the_stage3_protocol(tmp_path, monkeypatch):
    """A compatibility artifact pointing at some other protocol cannot license
    this comparison."""
    d, mod = build(tmp_path)
    doctored = tmp_path / "compat.json"
    payload = json.loads(mod.COMPAT_V2.read_text())
    payload["bound_to_historical_protocol"]["evaluation_protocol_hash"] = "0" * 64
    doctored.write_text(json.dumps(payload))
    monkeypatch.setattr(mod, "COMPAT_V2", doctored)
    assert mod.PhaseADriver.stage0(d) is False
    assert "v2 compatibility artifact is bound to" in d.ev["stages"]["0"]["reason"]


def test_the_historical_attestation_is_read_not_rewritten():
    """250f72ef is historical fact. The migration must not have touched it."""
    import subprocess
    for rel in ("logs/autoinit_stage3_complete/attested_evaluation_protocol.json",
                "logs/autoinit_stage3_complete/materialized_thresholds.json",
                "logs/autoinit_stage3_complete/engine_probe.json"):
        diff = subprocess.run(["git", "diff", "--", rel], cwd=REPO,
                              capture_output=True, text=True).stdout
        assert diff == "", f"{rel} was modified; it is historical evidence"
    att = json.loads((REPO / "logs/autoinit_stage3_complete"
                      / "attested_evaluation_protocol.json").read_text())
    assert att["evaluation_protocol_hash"] == STAGE3_HASH


def test_a_real_content_digest_is_material_in_full(tmp_path):
    """The other half of the rule, which no live evidence exercises yet.

    Every saved probe uses the `ref@driver` fallback because RunPod hosts do not
    expose `/etc/podinfo/image_digest`. The rule also says a REAL container image
    digest is material in full when it is available -- and a mutation making that
    branch treat `sha256:...` as a driver went undetected, because nothing ran
    it. Exercised directly here rather than waiting for a host that provides one.
    """
    from aadistill.autoinit.generation_compat import (
        ComparabilityError, comparable_generation_identity,
        require_comparable, split_image_identity,
    )

    a = split_image_identity("runpod/pytorch:1.0@sha256:" + "a" * 64)
    assert a["container_image_digest"] == "sha256:" + "a" * 64
    assert a["nvidia_driver_version"] is None
    assert a["form"] == "content_digest"

    protocol = json.loads(
        (REPO / "logs/autoinit_stage3_complete/attested_evaluation_protocol.json")
        .read_text())["evaluation_protocol"]
    base_runtime = json.loads(STAGE3_PROBE.read_text())["runtime"]

    def ident(image):
        return comparable_generation_identity(
            protocol=protocol, runtime={**base_runtime, "image_digest": image})

    one = ident("runpod/pytorch:1.0@sha256:" + "a" * 64)
    same = ident("runpod/pytorch:1.0@sha256:" + "a" * 64)
    other = ident("runpod/pytorch:1.0@sha256:" + "b" * 64)

    assert one["comparable_identity"] == same["comparable_identity"]
    # Two different CONTENT digests are two different images: not comparable.
    with pytest.raises(ComparabilityError):
        require_comparable(other, one, context="content digest")
    # And a content digest is not interchangeable with the fallback form.
    with pytest.raises(ComparabilityError):
        require_comparable(ident(STAGE3_IMAGE), one, context="form mismatch")


def test_an_unexpected_in_process_failure_keeps_its_traceback(tmp_path):
    """Attempt 6's stage 1 died in-process and left only `type: message`.

    Stages that shell out keep a log tail. Stages 1 and 5 do not, so the frame
    is gone unless the driver writes it down. The short reason must stay short —
    it is what the session record and the launcher print — and the frame must
    land in the audit evidence, in BOTH the JSON and a collected `.log`.
    """
    d, mod = build(tmp_path)
    d.results, d.ev = {}, {"stages": {}}

    def exploding_stage():
        def inner():
            raise RuntimeError("index is on cuda:0, different from other "
                               "tensors on cpu")
        inner()

    d.stage0 = exploding_stage
    for n in (1, 2, 3, 4, 5):
        setattr(d, f"stage{n}", lambda: True)
    d.finish = lambda success, failed: None

    rc = mod.PhaseADriver.run(d)
    assert rc == 20, "a blocking stage-0 failure must stop the run"

    entry = d.ev["stages"]["0"]
    assert entry["reason"].startswith("RuntimeError: index is on cuda:0"), (
        "the short reason changed; it is what the session record reports")
    assert "\n" not in entry["reason"], "the short reason must stay one line"

    frame = entry["traceback"]
    assert "Traceback (most recent call last)" in frame
    assert "in exploding_stage" in frame and "in inner" in frame, (
        "the frame that raised is missing; that is the whole point")

    written = (mod.AUDIT / entry["traceback_file"]).read_text()
    assert entry["traceback_file"] == "stage0_traceback.log"
    assert "in inner" in written
    assert written.startswith("stage 0: RuntimeError:")

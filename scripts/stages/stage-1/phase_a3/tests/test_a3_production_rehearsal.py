"""The real `A3Driver.run` control flow, end to end, at `$0`.

Not a canary. This drives the **actual** driver — its own `run`, its own stage
methods, its own gates — with only the hardware-bound operations replaced by
deterministic local fakes that emit the real filesystem layout and the real
file contracts. Everything between them is production code.

What it exists to prevent, in order of what has already cost this programme
money:

* an evaluation starting before all three trainings finish;
* a probe config drifting outside the frozen recipe's allowed override set;
* generations reaching the scorer without the protocol admission;
* an arm-vocabulary default swallowing A3's arm id after a complete
  measurement — the defect that ended attempt75 one stage from its verdict;
* the A-bsz3 digest being treated as a gate rather than a finding;
* a missing A-bsz1 incumbent gate, which would make the control reuse void;
* the launcher emitting a flag the driver's parser has no option for.

Every load-bearing rule is mutation-checked: the test asserts that breaking it
makes the test fail, because a suite that only ever passes is unverified.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
#: `tests/pod` is DELIBERATELY NOT on this list. Inserting it made `conftest`
#: resolve to tests/pod/conftest.py for every suite collected afterwards, and
#: tests/pod/test_phase_a_search_profile_seam.py stopped being able to import
#: `make_profile` from its own conftest -- a collection error in a suite this
#: round never touched. A test directory is not a package root to borrow from.
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "pod",
          REPO / "scripts" / "autoinit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stages.phase_a3 import autoinit_a3_driver as D  # noqa: E402

from stages.phase_c1.scoring import C1_BATTERY_SETS  # noqa: E402
from stages.phase_a3 import a3_session as A3S  # noqa: E402

BATTERY = REPO / "artifacts/stages/stage-1/phase_c1/batteries/c1_confirmation_v1"
SEEDS = list(A3S.recovery_seeds())

needs_battery = pytest.mark.skipif(
    not (BATTERY / "manifest.json").is_file(),
    reason=("the confirmation battery is not staged; the rehearsal drives the "
            "REAL scorer over REAL prompt ids and will not invent them"))


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, what: str) -> None:
        self.calls.append(what)


# --- shared fixture helpers, imported rather than re-written ---------------

#: One generated row, in the shape `uncapped_eval` writes.
GENERATION_RECORD = {
    "raw": "<think>\nreasoning.\n</think>\nThe answer is 42.",
    "think_preopened": True,
    "natural_termination": True,
    "degeneration_triggered": False,
    "context_limit_reached": False,
    "generated_tokens": 12,
    "stop_reason": "eos",
}

_SUMMARY_CACHE: dict = {}


def _summary_template() -> dict:
    """The per-set summary `uncapped_eval` writes, from a REAL retained one.

    `observe_generation_protocol` fails closed on any missing declared field,
    so modelling the fake on a plausible subset makes the PRODUCTION path
    fail -- which is the harness working and useless as a fixture. Read from a
    retained summary rather than imported from another test module: borrowing
    it required `tests/pod` on `sys.path`, which broke an unrelated suite's
    conftest resolution.
    """
    if _SUMMARY_CACHE:
        return dict(_SUMMARY_CACHE)
    from stages.phase_c1 import verify_c1_scoring_equivalence as EQ

    for d in EQ.find_generations():
        q = d / "gsm8k.json"
        if q.is_file():
            _SUMMARY_CACHE.update(json.loads(q.read_text()))
            return dict(_SUMMARY_CACHE)
    pytest.skip("no retained generation summary to model the fake on")


def _generation_record() -> dict:
    return dict(GENERATION_RECORD)


def _write_battery_generations(gen_dir: Path, *, drift: bool = False) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    record = _generation_record()
    for name in C1_BATTERY_SETS:
        ids = [json.loads(x)["id"]
               for x in (BATTERY / f"{name}.jsonl").open() if x.strip()]
        with (gen_dir / f"{name}.generations.jsonl").open("w") as fh:
            for i in ids:
                fh.write(json.dumps({**record, "id": i}) + "\n")
        summary = {**_summary_template(), "label": gen_dir.name,
                   "prompts": str(BATTERY / f"{name}.jsonl"),
                   "n_samples": len(ids)}
        if drift:
            #: One MATERIAL generation field moved. The engine still produced a
            #: complete set of rollouts; they were simply not produced under
            #: the protocol the session attested.
            summary["sampling"] = {**summary.get("sampling", {}),
                                   "temperature": 0.7}
        (gen_dir / f"{name}.json").write_text(json.dumps(summary))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Redirect every A3 root into tmp, and fake only the hardware."""
    rec = Recorder()
    audit, train, evald, work = (tmp_path / n for n in
                                 ("audit", "train", "eval", "work"))
    for attr, val in (("AUDIT", audit), ("TRAIN", train), ("EVAL", evald),
                      ("WORK", work), ("STATUS", tmp_path / "a3.status")):
        monkeypatch.setattr(D, attr, val)
    for d in (audit, audit / "probes", audit / "configs", train, evald, work):
        d.mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(rec=rec, tmp=tmp_path, audit=audit,
                                 train=train, eval=evald)


def _args(**over):
    a = D.build_parser().parse_args(
        ["--image-digest", "img@test", "--rate", "1.098333",
         "--soft-stop-usd", "8.1335", "--authorized-usd", "8.2525",
         "--run-id", "a3_rehearsal"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


class _FakeStep:
    def __init__(self, i, digest, tmp):
        self.index = i
        self.identity = types.SimpleNamespace(artifact_digest=digest)
        self.checkpoint_path = str(tmp / f"ckpt{i}")
        self.result_spec_hash = "s" * 64
        self.profile_id = "calib.domain_balanced@v1"
        self.impl_id = f"op{i}"
        self.trace = {}

    def as_dict(self):
        return {"index": self.index,
                "artifact_digest": self.identity.artifact_digest}


def _fake_hardware(monkeypatch, h, *, parent_digest=None, incumbent_digest=None,
                   a3_digest="a" * 64, train_fails=None, drift_at=None,
                   diagnostics_invalid=False):
    """Replace ONLY the hardware-bound operations. The contracts stay real."""
    rec = h.rec
    parent = parent_digest or A3S.expected_parent_digest()
    incumbent = incumbent_digest or A3S.expected_incumbent_digest()

    def stage_b(self):
        D.mark("STAGE_START:B")
        rec("teacher_verify")
        self.teacher_path = str(h.tmp / "teacher")
        self.complete("B", repo_id="fake", revision="rev")
    monkeypatch.setattr(D.A3Driver, "stage_b", stage_b)

    def gate(self, name, cmd, *, timeout, python=None):
        rec(f"gate:{name}")
        if name.startswith("score:"):
            return _run_real_scorer(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(D.A3Driver, "gate", gate)

    def release_device(self):
        rec("cuda_handoff")
        (D.AUDIT / "a3_device_handoff.json").write_text(
            json.dumps({"releases": [{"verdict": "released"}]}))
    monkeypatch.setattr(D.A3Driver, "release_device", release_device)

    def fake_materialize(spec, **kw):
        rec("replay")
        kw["root_loader"]()                       # the closure under test
        return [_FakeStep(0, "d" * 64, h.tmp), _FakeStep(1, "e" * 64, h.tmp),
                _FakeStep(2, parent, h.tmp), _FakeStep(3, incumbent, h.tmp)]
    monkeypatch.setattr(D, "materialize_fixed_path", fake_materialize)
    monkeypatch.setattr(D, "write_replay_record",
                        lambda spec, results, path, **kw: Path(path).write_text(
                            json.dumps({"steps": [r.as_dict() for r in results],
                                        "all_pinned_digests_matched": True})))
    monkeypatch.setattr(D, "require_headroom", lambda *a, **k: None)
    monkeypatch.setattr(D, "cuda_memory", lambda *a, **k: {"free": 1 << 40})

    from aadistill.initialization.specs.arch import get_adapter
    adapter = get_adapter("qwen3")
    monkeypatch.setattr(adapter, "load",
                        lambda *a, **k: types.SimpleNamespace(
                            config=types.SimpleNamespace()))

    def fake_structural(spec, **kw):
        rec("diagnostics")
        a3_dir = D.WORK / "diagnostics" / A3S.TREATMENT_PROTOCOL / "rep0"
        a3_dir.mkdir(parents=True, exist_ok=True)
        base = {"kept_q_heads_per_layer": [[0, 1]], "head_scores_per_layer": [[1.0, 2.0]],
                "selection_margin_per_layer": [[0.5]], "calibration_tokens": 59830,
                "result_spec_hash": "s" * 64, "peak_vram_gib": 3.5,
                "observed_execution_counters": {
                    "physical_forward_invocations": 67,
                    "executed_positions": 59830, "valid_positions": 59830,
                    "padded_positions": 0},
                "rounds": [{"round": 0, "warm_up": True}],
                "timing": {"scorer_seconds": {"n": 3, "mean": 11.2}}}
        comparison = {
            "artifact_digest_identical": incumbent == a3_digest,
            "classification": ("TRANSPARENT_EXECUTION_OPTIMIZATION"
                               if incumbent == a3_digest
                               else "DISTINCT_NUMERICAL_MATERIALIZATION_PROTOCOL"),
            "rounds_completed": 4,
            "incumbent_digest_gate": {"checked": True, "matches": True,
                                      "expected": incumbent,
                                      "observed": incumbent},
            "runtime": {"scorer_speedup_bsz1_over_bsz3": {"on_means": 1.4}},
            "kept_head_selection": {"identical": True, "slots_differing": 0},
            "head_scores": {"rank_correlation": {"overall": 0.99}},
            "execution_counters": {"masking_invariant_holds": True,
                                   "prediction_held": True},
        }
        if diagnostics_invalid:
            comparison["INVALID"] = ["the two protocols saw different tokens"]
        #: A REAL CHECKPOINT-SHAPED DIRECTORY per protocol, at the depth the
        #: materializer actually writes: `<workdir>/steps/03_attention`, not
        #: the round's workdir. The fake used to omit `checkpoint_path`
        #: entirely and the driver rebuilt the path by hand -- which is how
        #: stage F came to hand the trainer a directory with no config.json
        #: and die at `$0.58` with the whole structural result in hand. The
        #: fake now emits what the producer emits, so the rehearsal exercises
        #: the handoff instead of assuming it.
        out = {}
        for protocol, digest, mbs in (("A_bsz1", incumbent, 1),
                                      ("A_bsz3", a3_digest, 3)):
            ckpt = (h.tmp / "diagnostics" / protocol / "rep0" / "steps"
                    / "03_attention")
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "config.json").write_text(
                json.dumps({"model_type": "qwen3", "_fake": True}) + "\n")
            out[protocol] = {**base, "execution": {"micro_batch_size": mbs},
                             "artifact_digest": digest,
                             "checkpoint_path": str(ckpt)}
        out["_comparison"] = comparison
        return out
    from stages.phase_a3 import compare_a_bsz3
    monkeypatch.setattr(compare_a_bsz3, "structural_half", fake_structural)

    #: THE ONLY hardware step of stage F. The loop, the override check, the
    #: journal, the completion count and the three-probe guard stay REAL.
    def train_one(self, name, config):
        rec(f"train:{name}")
        trained = len([c for c in rec.calls if c.startswith("train:")])
        if train_fails is not None and trained == train_fails:
            raise D.A3DriverError(f"{name}: training failed rc=1")
        out_dir = D.TRAIN / name
        model = out_dir / "checkpoints" / "step_001023" / "model"
        model.mkdir(parents=True, exist_ok=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_text("weights")
        (out_dir / "train_log.jsonl").write_text('{"step":1}\n')
        (out_dir / "run_manifest.json").write_text("{}")
        (out_dir / "run_completion.json").write_text(
            json.dumps({"final_step": 1023, "config_sha256": "x"}))
        return out_dir
    monkeypatch.setattr(D.A3Driver, "train_one", train_one)

    def preserve_probe(self, name, model_dir, record):
        rec(f"preserve:{name}")
        return {"preserved": True, "relay_prefix": f"a3/{name}",
                "files": {"model.safetensors": "f" * 64}, "bytes": 1}
    monkeypatch.setattr(D.A3Driver, "preserve_probe", preserve_probe)

    def generate_one(self, name, package, gen_dir, sets):
        rec(f"generate:{name}")
        n = len([c for c in rec.calls if c.startswith("generate:")])
        _write_battery_generations(Path(gen_dir), drift=(drift_at == n))
    monkeypatch.setattr(D.A3Driver, "generate_one", generate_one)

    def attest(self):
        """Only the engine probe is faked. The protocol object is REAL.

        `admit_generation` compares against `self.evaluation_protocol`, so a
        placeholder here would make the comparison vacuous and the harness
        would certify a gate that never ran.
        """
        rec("engine_probe")
        clean = D.EVAL / "_attest_reference"
        _write_battery_generations(clean)
        summaries = [json.loads(q.read_text())
                     for q in sorted(clean.glob("*.json"))
                     if not q.name.endswith(".generations.jsonl")]
        gen = D.observe_generation_protocol(summaries).protocol
        manifest = json.loads((D.BATTERY / "manifest.json").read_text())
        contract = D.c1_scoring_contract(D.REPO)
        self.evaluation_protocol = D.RecoveryEvaluationProtocol(
            generation=gen, scoring_contract=contract["contract"],
            scoring_digest=contract["digest"],
            battery_artifact=manifest["artifact"],
            battery_manifest_sha256=manifest["manifest_sha256"],
            battery_content_sha256=manifest["content_sha256"])
        #: THE RUNTIME BLOCK the engine probe observes, which
        #: `admit_generation` needs for the live comparable identity. The fake
        #: omitted it and the driver crashed inside
        #: `comparable_generation_identity` on a None -- so the rehearsal now
        #: emits the production contract, and the driver refuses a missing
        #: runtime with a diagnosis instead of an AttributeError.
        #:
        #: The versions are the CONTROLS' own, read from their attestation, so
        #: the rehearsed comparison is the real one: material fields equal,
        #: driver patch within the branch.
        #: A HISTORICAL SIDE built from these same fake summaries, with a
        #: driver patch DELIBERATELY different inside the same branch. The
        #: rehearsal cannot reproduce attempt75's engine stack on a dev box,
        #: and pretending to would make the check vacuous; what it CAN prove
        #: is the property that matters -- the v2 rule tolerates a patch
        #: within a branch and refuses a material difference. a3_attempt35
        #: died because the real controls' branch (580) and the host's (595)
        #: differed, and `host_admission` now rejects such a host before setup.
        control = json.loads(D.CONTROL_ATTESTATION.read_text())
        #: FROM THE ENGINE PROBE, because that is what the driver compares
        #: against now: attempt75's attestation carries a TRAIN runtime
        #: (`cuda_runtime 12.8`) and generation runs under the vLLM venv
        #: (`13.0`), so taking the live side from the attestation made the
        #: runtime stacks differ for a reason neither side disagrees about.
        probe75 = json.loads(D.CONTROL_ENGINE_PROBE.read_text())
        live_runtime = dict(probe75.get("runtime") or probe75)
        #: A DIFFERENT PATCH inside the same branch, which is the property
        #: worth rehearsing: v2 demotes the patch and refuses a branch change.
        live_runtime["image_digest"] = (
            "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.126.09")
        self.observed_runtime = live_runtime
        fake_control = h.tmp / "fake_control_attestation.json"
        fake_control.write_text(json.dumps({
            "runtime": control["runtime"],
            "evaluation_protocol": self.evaluation_protocol.as_dict(),
        }) + "\n")
        monkeypatch.setattr(D, "CONTROL_ATTESTATION", fake_control)
        return {"evaluation_protocol_hash":
                    self.evaluation_protocol.evaluation_protocol_hash,
                "generation_protocol_fingerprint": gen.fingerprint}
    monkeypatch.setattr(D.A3Driver, "attest", attest)

    def build_package(model_dir, *, tokenizer_source, dest,
                      expected_sidecar_sha256):
        Path(dest).mkdir(parents=True, exist_ok=True)
        return {"tokenizer_source_rule": "the evaluated checkpoint"}
    monkeypatch.setattr(D, "build_evaluation_package", build_package)
    return rec


def _run_real_scorer(cmd):
    """Run the ACTUAL scorer as a subprocess, with this interpreter."""
    import subprocess

    argv = [sys.executable, *[str(c) for c in cmd[1:]]]
    env = {"PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}",
           "PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    return subprocess.run(argv, capture_output=True, text=True, timeout=1800,
                          env=env)


# --- the chain runs, end to end -------------------------------------------


@needs_battery
def test_the_whole_chain_runs_and_reaches_all_done(harness, monkeypatch):
    rec = _fake_hardware(monkeypatch, harness)
    rc = D.A3Driver(_args()).run()
    status = (harness.tmp / "a3.status").read_text()
    assert rc == 0, status
    assert "ALL_DONE" in status
    #: Every stage passed, in order, and none was skipped.
    for letter in A3S.STAGE_LETTERS:
        assert f"STAGE_PASSED:{letter}" in status, letter
    #: THERE IS NO STAGE I. A decision marker would mean the pod computed one.
    assert "STAGE_PASSED:I" not in status
    #: Three probes trained, three scored, in that order.
    trains = [c for c in rec.calls if c.startswith("train:")]
    gens = [c for c in rec.calls if c.startswith("generate:")]
    assert len(trains) == 3 and len(gens) == 3
    assert rec.calls.index(trains[-1]) < rec.calls.index(gens[0]), (
        "an evaluation began before the last training finished; the "
        "confirmation battery is met once per fully trained probe")
    #: And the evidence the OFF-POD comparison reads exists.
    for name in ("a3_evidence.json", "a3_diagnostics.json",
                 "a3_arm_identities.json", "a3_probe_inventory.json",
                 "a3_replay.json", "a3_device_handoff.json"):
        assert (harness.audit / name).is_file(), name
    inv = json.loads((harness.audit / "a3_probe_inventory.json").read_text())
    assert inv["n_probes"] == 3
    assert sorted(inv["seeds"]) == sorted(SEEDS)
    assert inv["arm"] == A3S.TREATMENT_ARM


@needs_battery
def test_a_differing_a_bsz3_digest_is_a_finding_and_the_chain_continues(
        harness, monkeypatch):
    """THE central behaviour of the 2026-10-01 redesign.

    The split design made a differing digest the TRIGGER for a second,
    separately approved study. A3 records it and trains.
    """
    differing = "f" * 64
    rec = _fake_hardware(monkeypatch, harness, a3_digest=differing)
    rc = D.A3Driver(_args()).run()
    status = (harness.tmp / "a3.status").read_text()
    assert rc == 0, status
    assert "A3_DIGESTS:DIFFERENT" in status
    assert "ALL_DONE" in status
    assert len([c for c in rec.calls if c.startswith("train:")]) == 3, (
        "a differing digest stopped the chain; it is a FINDING")
    ids = json.loads((harness.audit / "a3_arm_identities.json").read_text())
    assert ids["artifact_digest_identical"] is False
    assert ids["classification"] == "DISTINCT_NUMERICAL_MATERIALIZATION_PROTOCOL"
    #: And the asymmetry is recorded: one gated, one measured.
    assert ids["protocols"]["A_bsz1"]["gated"] is True
    assert ids["protocols"]["A_bsz3"]["gated"] is False
    assert ids["protocols"]["A_bsz3"]["expected_artifact_digest"] is None


@needs_battery
def test_identical_digests_also_run_the_whole_chain(harness, monkeypatch):
    """The other branch. A3 trains either way."""
    same = A3S.expected_incumbent_digest()
    rec = _fake_hardware(monkeypatch, harness, a3_digest=same)
    assert D.A3Driver(_args()).run() == 0
    status = (harness.tmp / "a3.status").read_text()
    assert "A3_DIGESTS:IDENTICAL" in status
    assert len([c for c in rec.calls if c.startswith("train:")]) == 3
    ids = json.loads((harness.audit / "a3_arm_identities.json").read_text())
    assert ids["classification"] == "TRANSPARENT_EXECUTION_OPTIMIZATION"


# --- and it refuses what it must refuse -----------------------------------


@needs_battery
def test_a_parent_that_does_not_reproduce_is_an_integrity_stop(
        harness, monkeypatch):
    _fake_hardware(monkeypatch, harness, parent_digest="0" * 64)
    rc = D.A3Driver(_args()).run()
    status = (harness.tmp / "a3.status").read_text()
    assert rc == 40
    assert "A3_REPLAY_MISMATCH" in status
    assert "STAGE_PASSED:D" not in status
    ev = json.loads((harness.audit / "a3_evidence.json").read_text())
    assert "parent digest" in ev["failure"]["error"]


@needs_battery
def test_an_a_bsz1_that_does_not_rebuild_the_incumbent_is_an_integrity_stop(
        harness, monkeypatch):
    """A-bsz1 IS canonical A. If it does not rebuild the incumbent, the
    attempt75 controls do not describe what this session built."""
    _fake_hardware(monkeypatch, harness, incumbent_digest="1" * 64)
    rc = D.A3Driver(_args()).run()
    status = (harness.tmp / "a3.status").read_text()
    assert rc == 40 and "A3_REPLAY_MISMATCH" in status
    ev = json.loads((harness.audit / "a3_evidence.json").read_text())
    assert "A-bsz1 IS canonical A" in ev["failure"]["error"]


@needs_battery
def test_void_diagnostics_stop_the_chain_rather_than_being_a_result(
        harness, monkeypatch):
    _fake_hardware(monkeypatch, harness, diagnostics_invalid=True)
    rc = D.A3Driver(_args()).run()
    status = (harness.tmp / "a3.status").read_text()
    assert rc == 40 and "A3_INTEGRITY_FAILURE" in status
    assert "STAGE_PASSED:E" not in status


@needs_battery
def test_a_drifted_generation_protocol_is_refused_before_the_scorer(
        harness, monkeypatch):
    """C2's confirmation set carried three fingerprints and produced no
    verdict. A probe generated under a drifted protocol is not scored, and no
    later probe is evaluated."""
    rec = _fake_hardware(monkeypatch, harness, drift_at=2)
    rc = D.A3Driver(_args()).run()
    assert rc == 40
    scored = [c for c in rec.calls if c.startswith("gate:score:")]
    assert len(scored) == 1, (
        "the drifted probe was scored, or a later probe was evaluated after "
        f"the refusal: {scored}")
    admissions = sorted(harness.audit.glob("*_generation_admission.json"))
    refused = [json.loads(p.read_text()) for p in admissions]
    assert any(r.get("comparable") is False for r in refused)


@needs_battery
def test_a_training_failure_stops_before_any_evaluation(harness, monkeypatch):
    rec = _fake_hardware(monkeypatch, harness, train_fails=2)
    rc = D.A3Driver(_args()).run()
    assert rc == 40
    assert not [c for c in rec.calls if c.startswith("generate:")], (
        "an evaluation ran after a training failure; stage G requires three "
        "completions")


@needs_battery
def test_the_probe_config_may_not_drift_from_the_frozen_recipe(
        harness, monkeypatch):
    """The recovery recipe is frozen science. Mutation-checked by widening
    the derived config beyond the allowed override set."""
    _fake_hardware(monkeypatch, harness)
    driver = D.A3Driver(_args())
    driver.a3_init_dir = harness.tmp / "init"
    driver.a3_init_digest = "a" * 64
    d = driver.descriptors()[0]
    #: The legitimate case passes.
    assert driver.probe_config(d).is_file()
    #: And a config that changed the LOSS is refused.
    real = D.json.loads
    monkeypatch.setattr(
        D, "A3_PROBE_OVERRIDES", frozenset({"run_name"}))
    with pytest.raises(D.A3DriverError, match="outside the allowed override"):
        driver.probe_config(d)


# --- the arm vocabulary that ended attempt75 ------------------------------


@needs_battery
def test_the_probe_record_carries_a3s_arm_and_not_c1s_roles(harness,
                                                            monkeypatch):
    """`C1ProbeRecord.allowed_arms` defaults to ['incumbent', 'treatment'].

    attempt75 passed C3's arm ids and raised `unknown arm 'A_incumbent'` in
    its aggregation, AFTER nine probes were trained, preserved and scored.
    A3 passes its vocabulary explicitly; this drives the real constructor.
    """
    from stages.phase_c1.probe_results import C1ProbeRecord, C1ResultsError

    #: VALID counts, so this test isolates the ARM check. An empty map trips
    #: the count guard first and the test would then pass for the wrong
    #: reason -- it would never reach the vocabulary it exists to check.
    from stages.phase_c1.probe_results import N_PROMPTS, N_SCORABLE

    common = dict(
        probe_id="autoinit.v1.phase_a3.A_bsz3.1", seed=1,
        initialization_artifact_digest="a" * 64,
        trained_run={}, result_path="r", result_sha256="b" * 64,
        per_sample_path="p", per_sample_sha256="c" * 64,
        generations={},
        counts={"n": N_PROMPTS, "usable": 500, "correct": 30,
                "n_scorable": N_SCORABLE, "usable_scorable": 480},
        rates={"correct_overall": 0.0353}, per_capability={},
        scoring_contract={}, battery={})
    #: With A3's vocabulary: accepted.
    C1ProbeRecord(arm=A3S.TREATMENT_ARM, allowed_arms=(A3S.TREATMENT_ARM,),
                  **common)
    #: With the DEFAULT: refused, which is exactly attempt75's failure.
    with pytest.raises(C1ResultsError, match="unknown arm"):
        C1ProbeRecord(arm=A3S.TREATMENT_ARM, **common)


def test_the_driver_passes_allowed_arms_at_every_construction_site():
    """Structural, so a new construction site cannot inherit the default."""
    src = Path(D.__file__).read_text()
    sites = src.count("C1ProbeRecord(")
    assert sites >= 1
    assert src.count("allowed_arms=") >= sites, (
        f"{sites} C1ProbeRecord construction site(s) and fewer "
        "`allowed_arms=` arguments; one of them inherits C1's two ROLES")


# --- the launcher/driver CLI seam -----------------------------------------


def test_the_launcher_command_parses_with_the_drivers_own_parser():
    """C1 attempt 7 cleared the pod test gate for the first time ever and then
    died at $0.4231 because the launcher emitted `--stage all` and argparse
    exited 2 before a line of the driver ran. The driver's parser is the
    authority on what it accepts, so it is what parses the string."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "a3lau_seam", REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_launch.py")
    launcher = importlib.util.module_from_spec(spec)
    sys.modules["a3lau_seam"] = launcher
    spec.loader.exec_module(launcher)

    #: The ACQUISITION LOOP's flags, not a subset. A subset is what let the
    #: launcher go without declaring `--scr`, `--session-commit` and
    #: `--bundle` -- three flags the loop sends and `SessionRunner` reads.
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/a3-rehearsal", "--run-id", "a3_attempt1",
         "--session-commit", "0" * 40, "--bundle", "aad_test.bundle",
         "--gpu", "NVIDIA L40S", "--max-price", "1.09"])
    args.disk_gb = 60
    session = launcher.spec(args)
    plan = session.budget.plan(price_per_hour=1.098333, authorized_usd=8.2525)
    ctx = types.SimpleNamespace(
        args=args, auth=types.SimpleNamespace(hard_cap_usd=8.2525),
        evidence={}, image_digest="img@1", price=1.09, spent_usd=0.42)
    cmd = launcher.driver_command(ctx, plan)
    argv = [a.strip("'") for a in cmd.split()[2:]]
    parsed = D.build_parser().parse_args(argv)
    assert parsed.authorized_usd == pytest.approx(8.2525)
    assert parsed.soft_stop_usd < parsed.authorized_usd
    assert parsed.run_id == "a3_attempt1"
    #: No stage-selection surface exists to emit.
    assert "--stage" not in cmd
    assert not any(a == "--stage" for a in argv)


# --- the resume-at-scoring path, through the real run() -------------------


def test_the_resume_path_scores_restored_probes_without_training(
        harness, monkeypatch):
    """The whole chain in resume mode: B, C, F-as-restore, G, H.

    No parent replay, no initialization, NO TRAINING -- and three scored
    endpoints at the end. This is AGENTS.md P8.4 state 2 made executable, and
    it exists because `a3_attempt36` spent `$3.39` retraining three probes
    that were already durable.

    The restore is faked at the DOWNLOAD only: `restore_one`'s digest
    verification, the five admission refusals in `preserved_records`, the
    descriptor rewrite and everything downstream are production code.
    """
    rec = _fake_hardware(monkeypatch, harness)

    #: A preserving attempt's records, in the shape the real ones have, with
    #: weights whose digests match what the record claims.
    source = "a3_attempt_source"
    records = (harness.tmp / "logs/stages/stage-1/phase_a3/runs" / source
               / "evidence" / "probes")
    records.mkdir(parents=True)
    blobs = harness.tmp / "relay"
    blobs.mkdir()
    init_digest = "7dd" + "0" * 61
    for seed in SEEDS:
        name = A3S.probe_id(seed)
        files = {}
        for filename, body in (("config.json", json.dumps({"model_type": "qwen3"})),
                               ("generation_config.json", "{}"),
                               ("model.safetensors", f"weights-{seed}")):
            b = blobs / f"{name}.{filename}"
            b.write_text(body)
            files[filename] = _sha256_text(body)
        (records / f"{name}.training.json").write_text(json.dumps({
            "probe_id": name, "arm": A3S.TREATMENT_ARM, "seed": seed,
            "initialization_artifact_digest": init_digest,
            "complete": True, "evaluated": False,
            "run_completion": "artifacts/stages/stage-3/a3/run_completion.json",
            "config_sha256": "c" * 64,
            "model_dir": f"artifacts/stages/stage-3/a3/{name}/checkpoints/step_001023/model",
            "preserved": {"preserved": True, "relay_repo": "r",
                          "relay_prefix": f"a3_preserved_probes/{source}/{name}",
                          "files": files},
        }) + "\n")
    monkeypatch.setattr(
        D, "RUNS_ROOT",
        harness.tmp / "logs/stages/stage-1/phase_a3/runs")
    import huggingface_hub

    def fake_download(*, repo_id, filename, **kw):
        name, leaf = filename.rsplit("/", 1)[-1], filename.rsplit("/", 1)[-1]
        probe = filename.split("/")[-2]
        return str(blobs / f"{probe}.{leaf}")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    rc = D.A3Driver(_args(resume_scoring_from=source)).run()
    status = (harness.tmp / "a3.status").read_text()

    assert rc == 0, status
    assert "ALL_DONE" in status, status
    #: The stages whose only consumer was the training did NOT run.
    assert "STAGE_START:D" not in status, "the parent was replayed anyway"
    assert "STAGE_START:E" not in status, "the initializations were rebuilt anyway"
    assert "engine_probe" in rec.calls, "stage G did not attest"
    assert not any("train:" in c for c in rec.calls), (
        f"a probe was TRAINED in resume mode: {rec.calls}")
    #: And three probes were scored. The recorder names a gated subprocess
    #: `gate:<name>`, so the prefix is the gate's.
    scored = [c for c in rec.calls if "score:" in c]
    assert len(scored) == 3, f"{len(scored)} probes scored, expected 3: {rec.calls}"
    for seed in SEEDS:
        journal = json.loads(
            (harness.audit / "probes" / f"{A3S.probe_id(seed)}.training.json"
             ).read_text())
        assert journal["restored_from"]["attempt"] == source
        assert journal["restored_from"]["_weights_were_not_retrained"] is True
        assert journal["evaluated"] is True


def _sha256_text(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()

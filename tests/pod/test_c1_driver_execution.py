"""The real `C1Driver.run` control flow, end to end, at `$0`.

Not a canary. This drives the **actual** driver — its own `run`, its own stage
methods, its own authorization and plan binding — with only the hardware-bound
operations replaced by deterministic local fakes that emit the real filesystem
layout and the real file contracts. Everything between them is production code.

What it is here to prevent, in order of what has already cost this project money:

* the driver silently inheriting Phase-A operational machinery again;
* an evaluation starting before all six trainings finish, which would let the
  first arm meet the confirmation battery before the last arm was trained;
* a probe reaching `recovery_search_v2` instead of `c1_confirmation_v1`;
* a replay mismatch emitting its marker without its evidence;
* the CUDA handoff gate being skipped before the trainer is spawned;
* a decision built from five results, or from unpaired prompts.

Every load-bearing failure is mutation-checked: the test asserts that breaking
the rule makes the test fail, because a suite that only ever passes is unverified.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

import autoinit_c1_driver as D  # noqa: E402

from aadistill.autoinit.c1_isolation import derive_recovery_seeds  # noqa: E402
from aadistill.autoinit.c1_probe_results import (  # noqa: E402
    C1ResultsError, decision_inputs,
)
from aadistill.autoinit.c1_scoring import C1_BATTERY_SETS  # noqa: E402

SEEDS = derive_recovery_seeds()
BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"


# --- the driver is C1's own -------------------------------------------------

def test_c1driver_is_not_a_phase_a_subclass():
    """Mutation target: restoring the inheritance must fail this."""
    names = [c.__name__ for c in D.C1Driver.__mro__]
    assert "PhaseADriver" not in names, names
    assert D.C1Driver.__mro__[1] is object


DRIVER_SRC = REPO / "scripts/pod/autoinit_c1_driver.py"


def _executable_text(path: Path) -> str:
    """Everything the module can actually DO: identifiers plus live strings.

    Prose is excluded deliberately. The first version of these checks grepped the
    raw file and failed on this driver's own docstring, which names
    `score_recovery_search.py` and `recovery_search_v2` precisely to explain why
    they are not used. A check that cannot tell an explanation from a call site
    would force the explanation to be deleted.
    """
    import ast

    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docstrings.add(d)
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.alias):
            parts.append(node.name)
            parts.append(node.asname or "")
        elif isinstance(node, ast.ImportFrom):
            parts.append(node.module or "")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                parts.append(node.value)
    return "\n".join(parts)


def test_the_driver_owns_run_and_imports_no_phase_a_operational_module():
    import ast

    assert "run" in vars(D.C1Driver)
    tree = ast.parse(DRIVER_SRC.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for forbidden in ("autoinit_phase_a_driver", "autoinit_phase_a_launch",
                      "phase_a_search", "aadistill.autoinit.phase_a"):
        assert forbidden not in imported, forbidden
    assert "PHASE_A_PLAN_V1" not in _executable_text(DRIVER_SRC)


def test_stage_h_can_only_reach_the_c1_scorer_and_the_c1_battery():
    code = _executable_text(DRIVER_SRC)
    assert "score_recovery_search" not in code
    assert "recovery_search_v2" not in code
    assert "c1_confirmation_v1" in code
    assert "score_c1_confirmation.py" in code


def test_all_c1_paths_are_c1_owned():
    assert D.AUDIT == REPO / "artifacts/audit/autoinit_c1"
    assert D.TRAIN == REPO / "artifacts/stage3/c1"
    assert D.EVAL == REPO / "artifacts/eval/c1"
    code = _executable_text(DRIVER_SRC)
    for bad in ("audit/autoinit_phase_a", "stage3/phase_a", "eval/phase_a"):
        assert bad not in code, bad


def test_the_trainer_headroom_is_derived_from_the_committed_measurement():
    """39.79 + 1.35 + 0.51 GiB. Not a written constant."""
    assert D._trainer_bytes() == int((39.79 + 1.35 + 0.51) * 2**30)
    src = (REPO / "scripts/pod/autoinit_c1_driver.py").read_text()
    assert "39.79" not in src.split("def _trainer_bytes")[1].split("return")[1]


# --- the harness ------------------------------------------------------------

class Recorder:
    """The order in which hardware-bound things were asked to happen."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, what: str) -> None:
        self.calls.append(what)

    def index(self, prefix: str) -> list[int]:
        return [i for i, c in enumerate(self.calls) if c.startswith(prefix)]


#: The `behavior_v0` record shape `uncapped_eval` writes. Restated from the
#: consumer's own reader (`usable_rollout._from_behavior_v0`) rather than
#: invented: a fake whose shape matches what the CONSUMER expects instead of what
#: the PRODUCER emits is how a defective line once got certified by an
#: end-to-end harness.
GENERATION_RECORD = {
    "raw": "<think>\nreasoning.\n</think>\nThe answer is 42.",
    "think_preopened": True,
    "natural_termination": True,
    "degeneration_triggered": False,
    "context_limit_reached": False,
    "generated_tokens": 12,
    "stop_reason": "eos",
}


def _write_battery_generations(gen_dir: Path) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    for name in C1_BATTERY_SETS:
        ids = [json.loads(x)["id"] for x in (BATTERY / f"{name}.jsonl").open()
               if x.strip()]
        with (gen_dir / f"{name}.generations.jsonl").open("w") as f:
            for i in ids:
                f.write(json.dumps({**GENERATION_RECORD, "id": i}) + "\n")
        (gen_dir / f"{name}.json").write_text(json.dumps(
            {**_summary_template(), "label": gen_dir.name,
             "prompts": str(BATTERY / f"{name}.jsonl"), "n_samples": len(ids)}))


#: The per-set summary `uncapped_eval` writes, taken from a REAL retained one so
#: `observe_generation_protocol` — which fails closed on any missing declared
#: field — runs for real in the harness. Writing a plausible subset instead made
#: the production path fail, which is the harness working.
_SUMMARY_CACHE: dict = {}


def _summary_template() -> dict:
    if _SUMMARY_CACHE:
        return _SUMMARY_CACHE
    import verify_c1_scoring_equivalence as EQ
    for d in EQ.find_generations():
        p = d / "gsm8k.json"
        if p.is_file():
            _SUMMARY_CACHE.update(json.loads(p.read_text()))
            return _SUMMARY_CACHE
    pytest.skip("no retained generation summary to model the fake on")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Redirect every C1 root into tmp, and fake only the hardware."""
    rec = Recorder()
    audit, train, evald, work = (tmp_path / n for n in
                                 ("audit", "train", "eval", "work"))
    for attr, val in (("AUDIT", audit), ("TRAIN", train), ("EVAL", evald),
                      ("WORK", work), ("STATUS", tmp_path / "c1.status")):
        monkeypatch.setattr(D, attr, val)
    for d in (audit, audit / "probes", audit / "configs", train, evald, work):
        d.mkdir(parents=True, exist_ok=True)

    # A structurally valid C1 authorization, at a scratch path outside the repo.
    from aadistill.autoinit.c1_authorization import (
        C1Authorization, c1_harness_digest, c1_hard_ceiling_usd, load_pricing,
    )
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "c1l_h", REPO / "scripts/pod/autoinit_c1_launch.py")
    launcher = importlib.util.module_from_spec(spec)
    sys.modules["c1l_h"] = launcher
    spec.loader.exec_module(launcher)
    auth_path = tmp_path / "auth.json"
    auth = C1Authorization(
        authorization_id="TEST-NOT-A-GRANT", granted_utc="2026-09-02T00:00:00Z",
        granted_by="regression harness", plan_id="autoinit.v1.phase_c1",
        plan_hash=launcher._plan_hash(),
        science_plan_hash=(
            "fb2eeea531f9f0d11f84b77cd47dff30697122de90a072a7a80c3a7535e89280"),
        expected_usd=float(load_pricing(REPO)["totals"]["expected_usd"]),
        hard_cap_usd=c1_hard_ceiling_usd(REPO),
        per_launch_hard_usd=c1_hard_ceiling_usd(REPO),
        authorized_stages=(0, 1, 2, 3, 4, 5),
        stage_conditions={}, scope_note="test",
        authorized_session_commit="0" * 40,
        harness_source_digest=c1_harness_digest(REPO)["digest"],
        provenance_commit="0" * 40)
    auth_path.write_text(json.dumps(auth.as_dict(), indent=1))
    monkeypatch.setattr(D.C1Driver, "AUTHORIZATION_PATH", str(auth_path))
    monkeypatch.setattr(D, "REPO", REPO)
    return types.SimpleNamespace(rec=rec, tmp=tmp_path, audit=audit,
                                 train=train, eval=evald)


def _args(**over):
    ap = D.build_parser()
    a = ap.parse_args(["--soft-stop-usd", "13.4277", "--authorized-usd", "13.7578"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _fake_hardware(monkeypatch, h, *, mismatch_at=None, train_fails=None,
                   handoff_fails=False):
    """Replace ONLY the hardware-bound operations. The contracts stay real."""
    rec = h.rec

    def stage_b(self):
        rec("teacher_verify")
        self.teacher_path = str(h.tmp / "teacher")
        self.complete("B", repo_id="fake", revision=D.CS.TEACHER_REVISION)
    monkeypatch.setattr(D.C1Driver, "stage_b", stage_b)

    class Step:
        def __init__(self, i, digest, expected, matches):
            self.index, self.digest_expected, self.digest_matches = i, expected, matches
            self.identity = types.SimpleNamespace(artifact_digest=digest)
            self.checkpoint_path = str(h.tmp / f"ckpt{i}")
            self.selection = {"step": i}
            self.impl_id = f"op{i}"
        def as_dict(self):
            return {"index": self.index, "artifact_digest":
                    self.identity.artifact_digest,
                    "digest_matches": self.digest_matches}

    def stage_de(self):
        D.mark("STAGE_START:D")
        rec("replay")
        runtime = {"gpu": "fake"}
        seen = []
        for i, (digest, expected) in enumerate([
                (None, None), (None, None),
                (D.CS.EXPECTED_PARENT_DIGEST, D.CS.EXPECTED_PARENT_DIGEST),
                (D.CS.EXPECTED_INCUMBENT_DIGEST, D.CS.EXPECTED_INCUMBENT_DIGEST)]):
            matches = None if expected is None else True
            if mismatch_at == i:
                digest, matches = "0" * 64, False
            step = Step(i, digest, expected, matches)
            seen.append(step)
            if matches is False:
                exc = D.FixedPathDigestMismatch(i, f"step{i}", expected, digest,
                                                {"steps": []})
                self.replay_mismatch(exc, runtime, seen)
                raise D.C1ReplayMismatch(str(exc))
            if expected is not None and matches:
                if i == 2:
                    self.complete("D", artifact_digest=digest)
                    D.mark("STAGE_START:E")
                else:
                    self.complete("E", artifact_digest=digest)
        (D.AUDIT / "c1_replay_record.json").write_text(json.dumps(
            {"schema": "aadistill.autoinit.fixed_path_replay/v1",
             "all_pinned_digests_matched": True, "runtime": runtime}, indent=1))
        self.parent, self.incumbent_step = seen[2], seen[3]
    monkeypatch.setattr(D.C1Driver, "stage_de", stage_de)

    def stage_f(self):
        D.mark("STAGE_START:F")
        rec("materialize_treatment")
        (D.AUDIT / "c1_arm_identities.json").write_text(json.dumps(
            {"schema": "aadistill.autoinit.c1_arm_identities/v1",
             "parent": self.parent.as_dict(), "shared_parent": True}, indent=1))
        self.arm_init = {"incumbent": (str(h.tmp / "inc"),
                                       D.CS.EXPECTED_INCUMBENT_DIGEST),
                         "treatment": (str(h.tmp / "trt"), "t" * 64)}
        self.complete("F", **{a: d for a, (_, d) in self.arm_init.items()})
    monkeypatch.setattr(D.C1Driver, "stage_f", stage_f)

    def release_device(self):
        rec("cuda_handoff")
        if handoff_fails:
            raise D.DeviceHandoffError("the card is still held")
        return {"handoff": {"verdict": "released"}, "need_bytes": D._trainer_bytes()}
    monkeypatch.setattr(D.C1Driver, "release_device", release_device)

    #: The ONLY hardware step of stage G. The loop, the override check, the
    #: journal, the completion count and the six-probe guard all stay REAL, so a
    #: mutation to any of them is visible here.
    def train_one(self, name, config):
        rec(f"train:{name}")
        trained = len([c for c in rec.calls if c.startswith("train:")])
        if train_fails is not None and trained == train_fails:
            raise D.C1DriverError(f"{name}: training failed rc=1")
        out_dir = D.TRAIN / name
        (out_dir / "checkpoints" / "t0" / "model").mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints" / "latest.txt").write_text("t0")
        (out_dir / "checkpoints" / "t0" / "model" / "config.json").write_text("{}")
        (out_dir / "train_log.jsonl").write_text('{"step":1}\n')
        (out_dir / "run_manifest.json").write_text("{}")
        (out_dir / "run_completion.json").write_text(
            json.dumps({"final_step": 1023, "config_sha256": "x"}))
        return out_dir
    monkeypatch.setattr(D.C1Driver, "train_one", train_one)

    def attest(self):
        rec("engine_probe")
        doc = {"generation_protocol_fingerprint": "f" * 64,
               "evaluation_protocol_hash": "e" * 64,
               "battery": {"content_sha256": D.C1_BATTERY_CONTENT_SHA256}}
        (D.AUDIT / "c1_attested_evaluation_protocol.json").write_text(
            json.dumps(doc, indent=2))
        return doc
    monkeypatch.setattr(D.C1Driver, "attest", attest)

    #: The ONLY hardware step of stage H. `require_all_trained`, the packaging,
    #: the REAL C1 scorer subprocess and the ordering all stay production code.
    def generate_one(self, name, package, gen_dir, sets):
        rec(f"evaluate:{name}")
        _write_battery_generations(Path(gen_dir))
    monkeypatch.setattr(D.C1Driver, "generate_one", generate_one)

    def build_package(model_dir, *, tokenizer_source, dest,
                      expected_sidecar_sha256):
        Path(dest).mkdir(parents=True, exist_ok=True)
        return {"tokenizer_source_rule": "the evaluated checkpoint"}
    monkeypatch.setattr(D, "build_evaluation_package", build_package)

    #: The fakes record ABSOLUTE paths, and `Path(repo) / "/abs"` is "/abs", so
    #: stage I's `REPO / path` resolves correctly without repointing REPO — which
    #: would break the REAL stage C, whose scoring-contract digest reads the
    #: repository.
    return rec


def _run(monkeypatch, h, **kw) -> tuple[int, D.C1Driver]:
    rec = _fake_hardware(monkeypatch, h, **kw)
    driver = D.C1Driver(_args())
    code = driver.run()
    return code, driver


# --- the happy path ---------------------------------------------------------

def test_the_full_session_runs_b_through_i_in_order(harness, monkeypatch):
    code, driver = _run(monkeypatch, harness)
    assert code == 0, driver.ev["stages"]
    assert driver.completed == [
        "teacher_fetch_verify", "register_operator", "replay_parent",
        "replay_incumbent", "materialize_arms", "recovery_probes", "evaluate",
        "decide"]
    assert driver.ev["outcome"] == "ALL_DONE"
    assert driver.ev["probes_trained"] == 6
    assert driver.ev["probes_evaluated"] == 6
    assert driver.ev["decision_ran"] is True
    assert driver.ev["formal_recovery_evidence"] == "OUT OF SCOPE"
    assert driver.ev["followon_started"] is False


def test_all_six_trainings_precede_the_first_evaluation(harness, monkeypatch):
    """The freshness of the confirmation battery depends on this ordering."""
    _run(monkeypatch, harness)
    calls = harness.rec.calls
    last_train = max(i for i, c in enumerate(calls) if c.startswith("train:"))
    first_eval = min(i for i, c in enumerate(calls) if c.startswith("evaluate:"))
    assert last_train < first_eval, calls
    assert len([c for c in calls if c.startswith("train:")]) == 6


def test_the_cuda_handoff_gate_precedes_every_training(harness, monkeypatch):
    _run(monkeypatch, harness)
    calls = harness.rec.calls
    assert calls.index("cuda_handoff") < min(
        i for i, c in enumerate(calls) if c.startswith("train:"))


def test_the_session_writes_every_report_the_launcher_fetches(harness, monkeypatch):
    _run(monkeypatch, harness)
    for name in ("c1_evidence.json", "c1_replay_record.json",
                 "c1_arm_identities.json", "c1_probe_results.json",
                 "c1_decision.json", "c1_attested_evaluation_protocol.json"):
        assert (harness.audit / name).is_file(), name


def test_the_decision_consumes_all_six_probes_and_the_real_rule(harness,
                                                                monkeypatch):
    _run(monkeypatch, harness)
    results = json.loads((harness.audit / "c1_probe_results.json").read_text())
    assert results["n_probes"] == 6
    assert len(results["probes"]) == 6
    assert sorted(results["seeds"]) == sorted(SEEDS)
    decision = json.loads((harness.audit / "c1_decision.json").read_text())
    assert decision["verdict"] in ("GO", "NO-GO", "INCONCLUSIVE")
    assert len(decision["per_seed_delta"]) == 3
    assert len(decision["mcnemar"]["per_seed"]) == 3
    assert decision["probe_results_sha256"] == results["results_sha256"]


def test_only_the_c1_battery_is_opened_in_stage_h(harness, monkeypatch):
    _run(monkeypatch, harness)
    for name in sorted(harness.audit.glob("*_c1_confirmation.json")):
        result = json.loads(name.read_text())
        assert result["battery"]["content_sha256"] == D.C1_BATTERY_CONTENT_SHA256
        assert result["scoring_contract"]["contract"] == "c1_confirmation_scoring@v1"
        assert result["n"] == 950 and result["n_scorable"] == 850


# --- the stop conditions ----------------------------------------------------

def test_a_replay_mismatch_trains_nothing_and_writes_its_evidence_first(
        harness, monkeypatch):
    code, driver = _run(monkeypatch, harness, mismatch_at=2)
    assert code == 30
    assert driver.ev["training_started"] is False
    assert not [c for c in harness.rec.calls if c.startswith("train:")]
    record = json.loads((harness.audit / "c1_replay_record.json").read_text())
    assert record["schema"] == "aadistill.autoinit.c1_replay_mismatch/v1"
    assert record["training_started"] is False
    status = (harness.tmp / "c1.status").read_text()
    assert status.index("C1_REPLAY_MISMATCH") > status.index("STAGE_FAILED:D")
    assert "STAGE_PASSED:D" not in status


def test_a_failed_handoff_stops_before_any_training(harness, monkeypatch):
    code, driver = _run(monkeypatch, harness, handoff_fails=True)
    assert code == 40
    assert not [c for c in harness.rec.calls if c.startswith("train:")]
    assert driver.ev["probes_trained"] == 0


def test_a_failed_fourth_training_produces_no_evaluation_and_no_decision(
        harness, monkeypatch):
    code, driver = _run(monkeypatch, harness, train_fails=4)
    assert code == 40
    assert not [c for c in harness.rec.calls if c.startswith("evaluate:")]
    assert driver.ev["probes_evaluated"] == 0
    assert driver.ev["decision_ran"] is False
    assert not (harness.audit / "c1_decision.json").exists()
    assert "C1_INCOMPLETE" in (harness.tmp / "c1.status").read_text()


# --- mutations --------------------------------------------------------------

def test_mutation_evaluating_before_six_trainings_is_refused(harness, monkeypatch):
    """Stage H's own guard, exercised directly."""
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.training = {("incumbent", SEEDS[0]): {}}
    with pytest.raises(D.C1DriverError, match="six training completions"):
        D.C1Driver.stage_h(driver)


def test_mutation_deciding_on_five_results_is_refused(harness, monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.scored = {("incumbent", s): {} for s in SEEDS}
    with pytest.raises(D.C1DriverError, match="not 6"):
        driver.stage_i()


def test_mutation_an_out_of_order_stage_is_refused(harness, monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.complete("B")
    with pytest.raises(Exception, match="out of order"):
        driver.complete("F")


def test_mutation_a_descriptor_without_its_identity_is_refused(harness,
                                                              monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.arm_init = {"incumbent": (str(harness.tmp), "i" * 64)}
    with pytest.raises(KeyError):
        driver.descriptors()


def test_mutation_a_probe_config_that_moves_a_frozen_field_is_refused(
        harness, monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.arm_init = {a: (str(harness.tmp / a), a * 32) for a in ("incumbent",
                                                                  "treatment")}
    d = driver.descriptors()[0]
    monkeypatch.setattr(D, "C1_PROBE_OVERRIDES", frozenset({"run_name"}))
    with pytest.raises(D.C1DriverError, match="outside the allowed override set"):
        driver.probe_config(d)


# --- the paired aggregation -------------------------------------------------

def _rows(n_scorable=850, ids=None, dup=False):
    rows = []
    for name, (n, domain, scorable) in C1_BATTERY_SETS.items():
        for i in range(n):
            rows.append({"id": f"{name}-{i}", "set": name, "domain": domain,
                         "scorable": scorable, "usable": True,
                         "correct": scorable and i % 7 == 0})
    if dup:
        rows.append(dict(rows[0]))
    return rows


def test_the_aggregation_refuses_a_duplicate_prompt_id():
    ps = {(a, s): _rows(dup=(a == "incumbent" and s == SEEDS[0]))
          for a in ("incumbent", "treatment") for s in SEEDS}
    with pytest.raises(C1ResultsError, match="duplicate prompt id"):
        decision_inputs(ps, seeds=SEEDS)


def test_the_aggregation_refuses_a_missing_probe():
    ps = {(a, s): _rows() for a in ("incumbent", "treatment") for s in SEEDS}
    ps.pop(("treatment", SEEDS[-1]))
    with pytest.raises(C1ResultsError, match="2 arms x 3 seeds"):
        decision_inputs(ps, seeds=SEEDS)


def test_the_aggregation_refuses_a_short_battery():
    ps = {(a, s): _rows() for a in ("incumbent", "treatment") for s in SEEDS}
    ps[("treatment", SEEDS[0])] = ps[("treatment", SEEDS[0])][:-1]
    with pytest.raises(C1ResultsError, match="949 prompts, not 950"):
        decision_inputs(ps, seeds=SEEDS)


def test_the_aggregation_uses_the_frozen_denominators_and_strata():
    ps = {(a, s): _rows() for a in ("incumbent", "treatment") for s in SEEDS}
    inputs = decision_inputs(ps, seeds=SEEDS)
    assert inputs.audit["n_prompts"] == 950
    assert inputs.audit["n_scorable"] == 850
    assert sum(inputs.audit["strata_sizes"].values()) == 850
    assert "code" not in inputs.audit["strata_sizes"]
    assert len(inputs.correct["incumbent"][SEEDS[0]]) == 850

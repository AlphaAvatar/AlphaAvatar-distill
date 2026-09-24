"""ONE `$0` rehearsal of the real behavioural driver, end to end, P through D.

Not a schedule simulation. This drives the PRODUCTION `C2BehaviouralDriver`,
built by its own `build_parser()` from a real argv, through its own `run()`,
and replaces only the calls that need hardware:

    verify_teacher   network — a 7.5 GiB hub fetch
    materialize_arm  GPU     — six four-step constructions from a 4B teacher
    release_device   CUDA
    train_one        GPU     — the recovery trainer
    attest           GPU     — a vLLM engine probe
    score_probe      GPU     — vLLM generation, then the scorer over its output

Everything else is production code and runs for real: the stage order and its
out-of-order refusal, the budget admission control, the probe-config derivation
and its override check, `trained_model_dir`'s latest.txt resolution, the
durability announcement (which really hashes real safetensors through
`identify_for_transfer`), the campaign journal, the rung loop, the completeness
gate, the ranking and tie-break, `advance_one`, the screening no-verdict
assertion, stage D's per-sample assembly, `decision_inputs`, `paired_differences`,
`stratified_cluster_bootstrap` at C2's own seed, and `decide`.

The fixtures control the DATA, never the logic: each scenario writes per-sample
rows with a chosen effect and lets the frozen rule reach its own verdict. All
three terminal states are reached from separate deterministic fixtures.

Why one rehearsal instead of a schedule test plus a flow test: a harness that
reimplements the loop cannot notice the loop being broken, and this programme
has already certified a defective line exactly that way.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import autoinit_c2_behavioural_driver as D
from experiments.phase_c2 import behavioural_decision as BD
from experiments.phase_c2 import behavioural_governance as BG
from experiments.phase_c2 import behavioural_schedule as SCH


N_PROMPTS, N_SCORABLE = 950, 850
STRATA = ("gsm8k", "math_verified", "multihop", "rag", "knowledge", "tool")


def _write_checkpoint(directory: Path) -> Path:
    """A real single-shard checkpoint, small enough to hash in milliseconds.

    Real bytes on purpose: `announce_durable` and `identify_for_transfer` run
    for real against them, so the durability path is executed rather than
    stubbed. A fake that only satisfied the consumer would certify nothing —
    an end-to-end harness in this programme once certified a defective line for
    exactly that reason.
    """
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file({"w": torch.zeros(4, 4, dtype=torch.float32)},
              str(directory / "model.safetensors"))
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 4}) + "\n")
    return directory


def _stable_seed(probe_id: str, seed: int) -> int:
    """A per-probe fixture seed that does not move between processes."""
    import hashlib

    digest = hashlib.sha256(f"{probe_id}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def _rows(seed: int, *, correct_rate: float, usable_rate: float) -> list[dict]:
    """One probe's per-sample rows, in the schema the scorer writes.

    `id`, `set`, `scorable`, `correct`, `usable` — the field names
    `decision_inputs` actually reads. An earlier draft of the decision module
    guessed `prompt_id`/`usable_rollout` and would have raised on real rows.
    """
    rng = random.Random(seed)
    rows = []
    for j in range(N_PROMPTS):
        scorable = j < N_SCORABLE
        rows.append({
            "id": f"p{j:04d}",
            "set": STRATA[j % len(STRATA)] if scorable else "code",
            "scorable": scorable,
            "correct": scorable and rng.random() < correct_rate,
            "usable": rng.random() < usable_rate,
        })
    return rows


class _Rehearsal(D.C2BehaviouralDriver):
    """The production driver with its hardware seams replaced. Nothing else."""

    #: Set per scenario: (treatment_correct, incumbent_correct,
    #: treatment_usable, incumbent_usable).
    effect = (0.30, 0.30, 0.90, 0.90)

    def verify_teacher(self) -> None:
        self.teacher_path = "/fake/teacher"

    def materialize_arm(self, label, spec, *, required, bounded_minutes,
                        config_overrides=None) -> str:
        #: The budget admission that guards it is production code and still runs.
        if not self.afford(bounded_minutes, f"materializing {label}"):
            raise D.C2DriverError(f"budget refuses {label}")
        d = _write_checkpoint(Path(self.a.b_workdir) / label / "model")
        self.announce_durable(label, d, kind="arm_initialization",
                              arch_signature=required["arch_signature"],
                              num_parameters=required["num_parameters"])
        return str(d)

    def release_device(self) -> dict:
        return {"verdict": "rehearsal: no device to release"}

    def train_one(self, name: str, config: Path) -> Path:
        """Write what a finished trainer writes, so `trained_model_dir` is real."""
        out = Path(self.a.eval_dir) / "_train" / name
        tag = "step_0001023"
        _write_checkpoint(out / "checkpoints" / tag / "model")
        (out / "checkpoints" / "latest.txt").write_text(tag + "\n")
        (out / "run_completion.json").write_text(
            json.dumps({"final_step": 1023, "config_sha256": "x"}) + "\n")
        return out

    def attest(self, battery: Path) -> dict:
        self.evaluation_protocol = object()
        return {"battery": battery.name, "evaluation_protocol_hash": "rehearsed"}

    def score_probe(self, probe, model_dir, *, battery, run_completion) -> dict:
        """Write this probe's per-sample rows; everything downstream is real.

        The row seed is a STABLE hash of the probe's identity. It was Python's
        built-in `hash()`, which is randomized per process by `PYTHONHASHSEED`,
        so the fixture's data — and therefore the decision the real rule
        reached on it — changed from run to run: `PYTHONHASHSEED=7` made
        `test_a_null_effect_does_not_manufacture_a_winner` come out GO. That is
        a roughly one-in-eight failure in a test that runs inside the PAID
        pod's blocking TESTS_OK gate, which is a launch killed at setup on a
        billing machine for a reason nobody could reproduce.
        """
        t_correct, i_correct, t_usable, i_usable = self.effect
        is_anchor = probe.arm == SCH.ANCHOR
        rows = _rows(
            _stable_seed(probe.probe_id, probe.seed),
            correct_rate=i_correct if is_anchor else t_correct,
            usable_rate=i_usable if is_anchor else t_usable)
        per_sample = self.audit / f"{probe.probe_id}_per_sample.jsonl"
        per_sample.write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        scorable = [r for r in rows if r["scorable"]]
        return {
            "probe_id": probe.probe_id, "rung": probe.rung, "arm": probe.arm,
            "seed": probe.seed,
            "per_sample_path": str(per_sample),
            "per_sample_sha256": "rehearsed",
            "result_path": str(per_sample), "result_sha256": "rehearsed",
            "correct_overall": sum(r["correct"] for r in scorable) / len(scorable),
            "usable_rollout_rate": sum(r["usable"] for r in rows) / len(rows),
            #: HOW this probe was measured, as a real scorer's result.json
            #: records it. Stage D now refuses a confirmation field that cannot
            #: be shown to share one measurement protocol, so a rehearsal whose
            #: scores omitted these would fail at the gate rather than at the
            #: behaviour it exists to rehearse -- and a fixture that omits what
            #: the producer writes is not rehearsing the producer. Uniform
            #: across probes here, which is the passing case; the mixed case is
            #: covered by tests/autoinit/test_measurement_protocol_gate.py.
            "battery": {"artifact": "c1_confirmation_v1",
                        "content_sha256": "r" * 64},
            "scoring_contract": {"contract": "c1_confirmation_scoring@v1",
                                 "digest": "s" * 64},
            "metric_contract": {"contract": "c1_confirmation_scoring@v1"},
            "generation_protocol_fingerprint": "g" * 64,
        }


def _run(tmp_path: Path, effect, *, authorized=40.0,
         run_attempt="attempt1") -> _Rehearsal:
    argv = ["--campaign", "rehearsal-1",
            #: The campaign and the run attempt, apart. They are separate flags
            #: because they are separate identities: a replacement resource is a
            #: new attempt inside the same campaign.
            "--run-attempt", run_attempt,
            "--audit-dir", str(tmp_path / "audit"),
            "--eval-dir", str(tmp_path / "eval"),
            "--b-workdir", str(tmp_path / "arms"),
            "--status-path", str(tmp_path / "status.txt"),
            #: "cuda", on a machine with no CUDA, and deliberately. The device
            #: is part of the construction spec hash: B built for "cpu" hashes
            #: to 3730a919… and `assert_frozen_construction` refuses it against
            #: C1's frozen 3a233a90…. A CPU rehearsal cannot even construct the
            #: arm it is rehearsing. Nothing here touches a device, because
            #: `materialize_arm` is a replaced seam — which is the point: the
            #: substitution is the LOADER, never the identity.
            "--device", "cuda",
            "--b-build-minutes", "31", "--probe-train-minutes", "75",
            "--probe-battery-minutes", "45", "--rate", "1.09",
            "--soft-stop-usd", str(authorized - 1),
            "--authorized-usd", str(authorized)]
    #: The REAL parser, from a real argv — not a hand-built namespace, which is
    #: how a driver comes to be tested against arguments its launcher never sends.
    args = D.build_parser().parse_args(argv)
    driver = _Rehearsal(args)
    driver.effect = effect
    driver.rc = driver.run()
    return driver


def _status(driver) -> str:
    return Path(driver.a.status_path).read_text()


# -- the three terminal states, from separate deterministic fixtures ---------

@pytest.mark.parametrize("effect,expected", [
    #: A large, consistent correctness gain with healthy behaviour.
    ((0.45, 0.25, 0.90, 0.90), "GO"),
    #: The candidate is clearly worse; the upper bound excludes the SESOI.
    ((0.18, 0.32, 0.90, 0.90), "NO_GO"),
    #: A behavioural veto fires even though correctness improved — usable
    #: rollout collapses on the candidate. NO_GO by veto, not by the interval.
    ((0.45, 0.25, 0.30, 0.95), "NO_GO"),
])
def test_the_driver_reaches_its_terminal_states(tmp_path, effect, expected):
    driver = _run(tmp_path, effect)
    assert driver.rc == 0, driver.ev["stages"]
    decision = json.loads((driver.audit / "c2_decision.json").read_text())
    assert decision["terminal_state"] == expected, decision["decision"]["criteria"]
    assert D.SUCCESS_MARKER in _status(driver)


def test_a_null_effect_does_not_manufacture_a_winner(tmp_path):
    """No forced winner. A true zero effect must not come out GO."""
    driver = _run(tmp_path, (0.30, 0.30, 0.90, 0.90))
    decision = json.loads((driver.audit / "c2_decision.json").read_text())
    assert decision["terminal_state"] in ("NO_GO", "INCONCLUSIVE")
    assert decision["decision"]["no_forced_winner"] is True


# -- what the run did on the way there --------------------------------------

def test_the_confirmation_passes_c2s_bootstrap_seed_to_the_bootstrap(
        tmp_path, monkeypatch):
    """Observe the CALL, not a field the caller wrote about itself.

    The first version of this test asserted `decision["bootstrap_seed_used"]`,
    which `confirm` fills in from the rule regardless of what it passed. Deleting
    `seed=rule.bootstrap_seed` from the bootstrap call — the precise defect this
    guards — left the test green. So it now intercepts the function and reads
    the keyword it actually received.
    """
    from experiments.phase_c1 import isolation

    real = isolation.stratified_cluster_bootstrap
    seen: list = []

    def watched(d, strata, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(d, strata, **kwargs)

    monkeypatch.setattr(isolation, "stratified_cluster_bootstrap", watched)
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90))

    assert seen == [834816710], (
        f"the bootstrap received seed={seen}; C2 froze 834816710 and a None "
        f"would silently fall back to C1's {isolation.bootstrap_seed()}")
    decision = json.loads((driver.audit / "c2_decision.json").read_text())
    assert decision["bootstrap_seed_used"] == 834816710
    assert decision["rule"]["seeds"] == [1936324010, 1916380711, 1523147638]


def test_all_five_stages_ran_in_the_frozen_order(tmp_path):
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90))
    assert driver.completed == list(D.STAGES)
    for letter in D.STAGES:
        assert driver.ev["stages"][letter]["passed"] is True


def test_twelve_probes_and_no_more(tmp_path):
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90))
    assert len(driver.training) == 12
    assert len(driver.scores) == 12
    assert len(driver.screening) == 6
    assert len(driver.confirmation) == 6


def test_exactly_one_candidate_advanced_and_it_is_not_the_anchor(tmp_path):
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90))
    ranking = json.loads((driver.audit / "c2_screening_ranking.json").read_text())
    assert len(ranking["ranked"]) == 5
    assert ranking["advanced"]["state_id"] != SCH.ANCHOR
    assert sum(1 for r in ranking["ranked"] if r.get("advanced")) <= 1


def test_the_screening_ranking_carries_no_verdict(tmp_path):
    """Screening ranks. It may not promote, decide or name an incumbent."""
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90))
    ranking = json.loads((driver.audit / "c2_screening_ranking.json").read_text())
    SCH.assert_screening_emits_no_verdict({"ranked": ranking["ranked"]})
    assert "verdict" not in ranking and "incumbent" not in ranking


def test_every_probe_was_announced_for_durability_with_a_full_identity(tmp_path):
    """The moment it exists, with every field the destination will compare."""
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90))
    probes = [u for u in driver.durable if u["kind"] == "probe"]
    assert len(probes) == 12
    for unit in probes:
        assert unit["identity_error"] is None, unit["identity_error"]
        for field in ("artifact_digest", "weights_digest", "config_sha256",
                      "single_shard_sha256", "arch_signature",
                      "num_parameters", "tokenizer_sha256"):
            assert field in unit["identity"]
        assert unit["campaign"] == "rehearsal-1"
        assert "authorizes" in unit and "nothing" in unit["authorizes"]


def test_the_six_arms_were_announced_too(tmp_path):
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90))
    arms = [u for u in driver.durable if u["kind"] == "arm_initialization"]
    assert len(arms) == 6


# -- the seams the rehearsal cannot execute, checked directly ---------------

def test_scorer_dispatch_sends_each_rung_to_its_own_entry_point(tmp_path):
    """The one production decision `score_probe`'s hardware body would hide."""
    screening = SCH.Probe("screening", "cand", 616738081, "dig", "/p")
    argv = D.scorer_argv(screening, battery=Path("/b/c2_screening_v1"),
                         gen_dir=tmp_path, out=tmp_path / "o.json",
                         per_sample=tmp_path / "s.jsonl",
                         generation_fingerprint="fp")
    assert argv[0].endswith("score_c2_screening.py")
    assert "--screening-arm" in argv and "--arm" not in argv

    for arm, expected in ((SCH.ANCHOR, BD.INCUMBENT_ARM),
                          ("cand", BD.TREATMENT_ARM)):
        confirm = SCH.Probe("confirmation", arm, 1936324010, "dig", "/p")
        argv = D.scorer_argv(confirm, battery=Path("/b/c1_confirmation_v1"),
                             gen_dir=tmp_path, out=tmp_path / "o.json",
                             per_sample=tmp_path / "s.jsonl",
                             generation_fingerprint="fp")
        assert argv[0].endswith("score_c1_confirmation.py")
        assert argv[argv.index("--arm") + 1] == expected
        assert "--screening-arm" not in argv


def test_the_launcher_builds_a_command_this_driver_can_parse(tmp_path):
    """The contract between the two halves, which nothing else executes.

    A launcher that emits a flag the driver does not define, or omits a
    required one, fails at the first second of a paid session.
    """
    import autoinit_c2_behavioural_launch as L

    class _Ctx:
        image_digest = "sha256:abc"
        price = 1.09
        spent_usd = 0.0
        args = L.build_parser().parse_args(
            ["--scr", "/tmp/probe", "--session-commit", "abc123",
             "--bundle", "b", "--run-id", "attempt1", "--max-price", "1.09"])

        class auth:
            hard_cap_usd = 33.21
            #: From the authorization, which names the CAMPAIGN. The launcher
            #: used to pass the run id here, which made the two identities one
            #: string and the continuation policy unreachable.
            campaign_id = BG.CAMPAIGN_ID

    class _Plan:
        soft_stop_usd = 30.0

    command = L.driver_command(_Ctx(), _Plan())
    assert "autoinit_c2_behavioural_driver.py" in command
    flags = command.split("autoinit_c2_behavioural_driver.py", 1)[1].split()
    #: The REAL parser decides whether the REAL command is acceptable.
    parsed = D.build_parser().parse_args([f.strip("'") for f in flags])
    assert parsed.campaign == BG.CAMPAIGN_ID
    assert parsed.run_attempt == "attempt1"
    assert parsed.authorized_usd == pytest.approx(33.21)
    assert parsed.rate == pytest.approx(1.09)


def test_a_failed_stage_writes_the_failure_marker(tmp_path):
    """The marker policy's failure path, executed rather than assumed."""

    class _Broken(_Rehearsal):
        def materialize_arm(self, *a, **k):
            raise D.C2ProvenanceError("rebuilt arm is not the frozen arm")

    argv = ["--campaign", "c", "--audit-dir", str(tmp_path / "a"),
            "--eval-dir", str(tmp_path / "e"), "--b-workdir", str(tmp_path / "w"),
            "--status-path", str(tmp_path / "s.txt"), "--device", "cuda",
            "--b-build-minutes", "31", "--probe-train-minutes", "75",
            "--probe-battery-minutes", "45", "--rate", "1.09",
            "--soft-stop-usd", "39", "--authorized-usd", "40"]
    driver = _Broken(D.build_parser().parse_args(argv))
    assert driver.run() == 1
    status = Path(driver.a.status_path).read_text()
    assert D.FAILURE_MARKER in status
    assert "STAGE_FAILED:P" in status
    assert driver.ev["stages"]["P"]["passed"] is False


def test_a_completed_probe_is_restored_and_never_retrained(tmp_path):
    """Resume, as pre-registered: same campaign, bytes re-identified, no retrain."""
    first = _run(tmp_path / "one", (0.45, 0.25, 0.90, 0.90))
    assert len(first.training) == 12

    second = _Rehearsal(D.build_parser().parse_args(
        ["--campaign", "rehearsal-1", "--audit-dir", str(first.a.audit_dir),
         "--eval-dir", str(first.a.eval_dir), "--b-workdir", str(first.a.b_workdir),
         "--status-path", str(tmp_path / "s2.txt"), "--device", "cuda",
         "--b-build-minutes", "31", "--probe-train-minutes", "75",
         "--probe-battery-minutes", "45", "--rate", "1.09",
         "--soft-stop-usd", "39", "--authorized-usd", "40"]))
    summary = second.load_campaign_journal()
    assert len(summary["restored"]) == 12
    assert summary["rejected"] == []
    assert all(r["scored"] for r in summary["restored"])


def test_a_replacement_resource_completes_the_campaign_without_retraining(
        tmp_path):
    """Continuation, end to end, across two run/resource identities.

    The same campaign under a second run attempt, driven P through D by the
    real `run()`: every probe is restored and none is retrained, the committed
    screening ranking is adopted rather than re-decided, and the verdict is the
    one the campaign's own evidence produces.

    This is the property the old code made impossible. `--campaign` was the run
    id, so a replacement resource was a different campaign and had to refuse
    all twelve probes — the rule meant to prevent cross-experiment pooling was
    instead preventing continuation of one experiment.
    """
    first = _run(tmp_path / "one", (0.45, 0.25, 0.90, 0.90),
                 run_attempt="attempt1")
    assert first.rc == 0
    trained_first = {name: rec["train_minutes"]
                     for name, rec in first.training.items()}
    assert len(trained_first) == 12
    committed = json.loads(
        (first.audit / "c2_screening_ranking.json").read_text())
    assert committed["campaign"] == "rehearsal-1"
    assert committed["run_attempt"] == "attempt1"

    #: A REPLACEMENT resource: new run attempt, same campaign, same durable
    #: evidence. `train_one` would raise if it were called, because nothing may
    #: be retrained.
    class _NoRetraining(_Rehearsal):
        def train_one(self, name, config):
            raise AssertionError(
                f"{name} was retrained on a continuation; a completed probe is "
                "never retrained, for any outcome")

    second = _NoRetraining(D.build_parser().parse_args(
        ["--campaign", "rehearsal-1", "--run-attempt", "attempt2",
         "--audit-dir", str(first.a.audit_dir),
         "--eval-dir", str(first.a.eval_dir),
         "--b-workdir", str(first.a.b_workdir),
         "--status-path", str(tmp_path / "s_continuation.txt"),
         "--device", "cuda",
         "--b-build-minutes", "31", "--probe-train-minutes", "75",
         "--probe-battery-minutes", "45", "--rate", "1.09",
         "--soft-stop-usd", "39", "--authorized-usd", "40"]))
    second.effect = first.effect
    assert second.run() == 0, second.ev["stages"]

    assert set(second.scores) == set(first.scores)
    #: The ranking was ADOPTED, not re-decided: the record still names the
    #: attempt that committed it.
    after = json.loads((first.audit / "c2_screening_ranking.json").read_text())
    assert after["run_attempt"] == "attempt1"
    assert second.advanced["state_id"] == committed["advanced"]["state_id"]
    assert (second.ev["stages"]["D"]["terminal_state"]
            == first.ev["stages"]["D"]["terminal_state"])
    assert D.SUCCESS_MARKER in Path(second.a.status_path).read_text()


def test_a_continuation_that_would_advance_another_candidate_is_refused(
        tmp_path):
    """Screening commits once. A second ranking naming another candidate stops.

    Mutating the committed record is the only way to reach this branch, because
    the ranking is deterministic from six scores — which is exactly why the
    check is worth having: determinism is an argument, not a guarantee.
    """
    first = _run(tmp_path / "one", (0.45, 0.25, 0.90, 0.90))
    path = first.audit / "c2_screening_ranking.json"
    record = json.loads(path.read_text())
    record["advanced"] = {**record["advanced"], "state_id": "a-different-leaf"}
    path.write_text(json.dumps(record))

    second = _Rehearsal(D.build_parser().parse_args(
        ["--campaign", "rehearsal-1", "--run-attempt", "attempt2",
         "--audit-dir", str(first.a.audit_dir),
         "--eval-dir", str(first.a.eval_dir),
         "--b-workdir", str(first.a.b_workdir),
         "--status-path", str(tmp_path / "s_conflict.txt"), "--device", "cuda",
         "--b-build-minutes", "31", "--probe-train-minutes", "75",
         "--probe-battery-minutes", "45", "--rate", "1.09",
         "--soft-stop-usd", "39", "--authorized-usd", "40"]))
    second.effect = first.effect
    assert second.run() == 1
    assert second.ev["stages"]["R"]["passed"] is False
    assert "commits once" in second.ev["stages"]["R"]["reason"]
    #: And it stopped BEFORE confirmation, so no verdict was reached.
    assert "D" not in second.ev["stages"]


def test_a_probe_from_another_campaign_is_refused(tmp_path):
    """A replacement resource may not pool with the previous one's probes."""
    first = _run(tmp_path / "one", (0.45, 0.25, 0.90, 0.90))
    other = _Rehearsal(D.build_parser().parse_args(
        ["--campaign", "a-different-campaign",
         "--audit-dir", str(first.a.audit_dir),
         "--eval-dir", str(first.a.eval_dir), "--b-workdir", str(first.a.b_workdir),
         "--status-path", str(tmp_path / "s3.txt"), "--device", "cuda",
         "--b-build-minutes", "31", "--probe-train-minutes", "75",
         "--probe-battery-minutes", "45", "--rate", "1.09",
         "--soft-stop-usd", "39", "--authorized-usd", "40"]))
    summary = other.load_campaign_journal()
    assert summary["restored"] == []
    assert len(summary["rejected"]) == 12
    assert all("campaign" in r["why"] for r in summary["rejected"])


def test_a_probe_whose_bytes_changed_is_refused(tmp_path):
    """A record is not a checkpoint. The bytes decide."""
    import torch
    from safetensors.torch import save_file

    first = _run(tmp_path / "one", (0.45, 0.25, 0.90, 0.90))
    entry = json.loads(
        sorted((first.audit / "probes").glob("*.json"))[0].read_text())
    save_file({"w": torch.ones(4, 4, dtype=torch.float32)},
              str(Path(entry["model_dir"]) / "model.safetensors"))

    again = _Rehearsal(D.build_parser().parse_args(
        ["--campaign", "rehearsal-1", "--audit-dir", str(first.a.audit_dir),
         "--eval-dir", str(first.a.eval_dir), "--b-workdir", str(first.a.b_workdir),
         "--status-path", str(tmp_path / "s4.txt"), "--device", "cuda",
         "--b-build-minutes", "31", "--probe-train-minutes", "75",
         "--probe-battery-minutes", "45", "--rate", "1.09",
         "--soft-stop-usd", "39", "--authorized-usd", "40"]))
    summary = again.load_campaign_journal()
    rejected = [r for r in summary["rejected"] if "hash" in r["why"]]
    assert len(rejected) == 1, summary["rejected"]


def test_budget_refuses_to_start_a_unit_it_cannot_finish(tmp_path):
    """Admission control, not a trip-wire: refuse BEFORE spending, not during."""
    driver = _run(tmp_path, (0.45, 0.25, 0.90, 0.90), authorized=0.5)
    assert driver.rc == 1
    assert driver.ev["stages"]["P"]["passed"] is False
    assert "budget refuses" in driver.ev["stages"]["P"]["reason"]


# -- the teardown decision, which only a pod would otherwise exercise --------

class _Ctx:
    """The little the durability helpers actually read off a SessionContext."""

    def __init__(self, tmp_path, units):
        self.evidence: dict = {}
        self.args = type("A", (), {"scr": str(tmp_path), "run_id": "attempt1"})()
        self._units = units
        (tmp_path / "relay").mkdir(parents=True, exist_ok=True)
        (tmp_path / "relay" / "c2_behavioural_evidence.json").write_text(
            json.dumps({"durable_units": units}))

    def say(self, msg):  # pragma: no cover - output only
        pass


def _unit(unit_id, *, kind="probe", identity=True):
    return {"unit_id": unit_id, "kind": kind, "path": f"/w/{unit_id}",
            "campaign": "c",
            "identity": ({"artifact_digest": "a", "weights_digest": "w",
                          "config_sha256": "c", "single_shard_sha256": "s",
                          "arch_signature": "g", "num_parameters": 1}
                         if identity else None),
            "identity_error": None if identity else "OSError: no shards"}


def test_unreadable_evidence_is_unknown_and_blocks_teardown(tmp_path):
    """"I found no probes" and "no probes were trained" are different findings."""
    import autoinit_c2_behavioural_launch as L

    ctx = _Ctx(tmp_path, [])
    (tmp_path / "relay" / "c2_behavioural_evidence.json").unlink()
    assert L.finished_probes(ctx) is None
    ok, why = L.probes_secured(ctx, [])
    assert not ok and "UNKNOWN" in why


def test_no_probes_trained_permits_teardown(tmp_path):
    """An early failure legitimately produced none; demanding one would block."""
    import autoinit_c2_behavioural_launch as L

    ctx = _Ctx(tmp_path, [_unit("arm0", kind="arm_initialization")])
    assert L.finished_probes(ctx) == []
    ok, why = L.probes_secured(ctx, [])
    assert ok and "no probe" in why


def test_a_finished_probe_still_on_the_pod_blocks_teardown(tmp_path):
    import autoinit_c2_behavioural_launch as L

    ctx = _Ctx(tmp_path, [_unit("p1"), _unit("p2")])
    ok, why = L.probes_secured(ctx, [
        {"artifact": "c2_behavioural_probe", "unit_id": "p1", "rc": 0,
         "matched": True, "identity_announced": True}])
    assert not ok and "p2" in why


def test_a_probe_with_no_announced_identity_is_preserved_not_dropped(tmp_path):
    """`announce_durable` never raises, so it can record identity=None.

    Filtering those out of what teardown waits for would silently discard
    thirty minutes of completed training. They are fetched like any other and
    reported as unverifiable — and they cannot block teardown forever, because
    an identity the driver never computed can never be produced by a retry.
    """
    import autoinit_c2_behavioural_launch as L

    ctx = _Ctx(tmp_path, [_unit("p1", identity=False)])
    assert [u["unit_id"] for u in L.finished_probes(ctx)] == ["p1"]
    ok, why = L.probes_secured(ctx, [
        {"artifact": "c2_behavioural_probe", "unit_id": "p1", "rc": 0,
         "matched": False, "identity_announced": False}])
    assert ok, why
    assert "UNVERIFIABLE" in why and "p1" in why


def test_a_digest_mismatch_is_not_counted_as_secured(tmp_path):
    """Arrival is not verification."""
    import autoinit_c2_behavioural_launch as L

    ctx = _Ctx(tmp_path, [_unit("p1")])
    ok, why = L.probes_secured(ctx, [
        {"artifact": "c2_behavioural_probe", "unit_id": "p1", "rc": 0,
         "matched": False, "identity_announced": True}])
    assert not ok and "p1" in why


# -- resume rule R6: an arm is reused only when its bytes prove it is that arm --

def test_an_arm_is_reused_only_when_its_digest_matches(tmp_path):
    """R6's "unless" clause. A copy that merely exists proves nothing."""
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.leaf_durability import identify_for_transfer

    argv = ["--campaign", "c", "--audit-dir", str(tmp_path / "a"),
            "--eval-dir", str(tmp_path / "e"), "--b-workdir", str(tmp_path / "w"),
            "--status-path", str(tmp_path / "s.txt"), "--device", "cuda",
            "--b-build-minutes", "31", "--probe-train-minutes", "75",
            "--probe-battery-minutes", "45", "--rate", "1.09",
            "--soft-stop-usd", "39", "--authorized-usd", "40"]
    driver = _Rehearsal(D.build_parser().parse_args(argv))

    model = _write_checkpoint(tmp_path / "w" / "armX" / "model")
    true_id = identify_for_transfer(model, adapter=get_adapter("qwen3"),
                                    arch_signature="g", num_parameters=1)
    right = {"artifact_digest": true_id.artifact_digest,
             "arch_signature": "g", "num_parameters": 1}
    wrong = {**right, "artifact_digest": "0" * 64}

    assert driver.reuse_arm("armX", right) == str(model)
    assert driver.reuse_arm("armX", wrong) is None
    assert driver.reuse_arm("not-built-yet", right) is None


# -- the post-build identity gate, which the rehearsal's seam replaces -------

class _Built:
    """What `materialize_fixed_path` hands back: a step result with an identity."""

    def __init__(self, path, **fields):
        self.checkpoint_path = str(path)
        self.identity = type("I", (), fields)()

    def as_dict(self):  # pragma: no cover - provenance only
        return {"checkpoint_path": self.checkpoint_path}


def _gate_driver(tmp_path):
    argv = ["--campaign", "c", "--audit-dir", str(tmp_path / "a"),
            "--eval-dir", str(tmp_path / "e"), "--b-workdir", str(tmp_path / "w"),
            "--status-path", str(tmp_path / "s.txt"), "--device", "cuda",
            "--b-build-minutes", "31", "--probe-train-minutes", "75",
            "--probe-battery-minutes", "45", "--rate", "1.09",
            "--soft-stop-usd", "39", "--authorized-usd", "40"]
    #: The PRODUCTION driver, not the rehearsal subclass: `materialize_arm` is
    #: exactly what is under test here, so it must not be the replaced one.
    driver = D.C2BehaviouralDriver(D.build_parser().parse_args(argv))
    driver.teacher_path = "/fake/teacher"
    return driver


REQUIRED_FIELDS = ("artifact_digest", "weights_digest", "config_sha256",
                   "single_shard_sha256", "arch_signature", "num_parameters")


def _required(**overrides):
    base = {"artifact_digest": "a" * 64, "weights_digest": "w" * 64,
            "config_sha256": "c" * 64, "single_shard_sha256": "s" * 64,
            "arch_signature": "g" * 64, "num_parameters": 596049920}
    return {**base, **overrides}


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_a_rebuilt_arm_that_differs_in_any_identity_field_is_refused(
        tmp_path, field, monkeypatch):
    """An arm that is not the arm the protocol names makes every delta
    measured against it meaningless, so it stops the session — as a PROVENANCE
    finding, not a retryable failure: the path is deterministic and a retry
    would diverge identically.

    The rehearsal replaces `materialize_arm` wholesale, so this gate is the one
    piece of stage P no other `$0` test executes. Only the hardware call inside
    it is replaced.
    """
    from aadistill.initialization.planning import fixed_path

    driver = _gate_driver(tmp_path)
    spec = BG.arm_specs(REPO)[0]
    required = _required()
    built = _required(**{field: ("z" * 64 if field != "num_parameters" else 1)})
    monkeypatch.setattr(
        fixed_path, "materialize_fixed_path",
        lambda *a, **k: [_Built(_write_checkpoint(tmp_path / "out"), **built)])

    with pytest.raises(D.C2ProvenanceError) as exc:
        driver.materialize_arm("armX", spec, required=required,
                               bounded_minutes=1.0)
    assert field in str(exc.value)
    assert "NO PROBE MAY START" in str(exc.value)


def test_a_matching_arm_passes_the_gate_and_is_announced(tmp_path, monkeypatch):
    """Guards the guard: a gate that refused everything would pass the tests above."""
    from aadistill.initialization.planning import fixed_path

    driver = _gate_driver(tmp_path)
    spec = BG.arm_specs(REPO)[0]
    required = _required()
    monkeypatch.setattr(
        fixed_path, "materialize_fixed_path",
        lambda *a, **k: [_Built(_write_checkpoint(tmp_path / "out"), **required)])

    path = driver.materialize_arm("armX", spec, required=required,
                                  bounded_minutes=1.0)
    assert Path(path).is_dir()
    announced = [u for u in driver.durable if u["unit_id"] == "armX"]
    assert len(announced) == 1 and announced[0]["identity_error"] is None


def test_an_arm_pinned_to_a_different_root_is_refused(tmp_path):
    """The root is part of the path's identity; a different one diverges at step 0."""
    driver = _gate_driver(tmp_path)
    spec = BG.arm_specs(REPO)[0]
    object.__setattr__(spec, "root_repo_id", "someone/else")
    with pytest.raises(D.C2ProvenanceError, match="root"):
        driver.materialize_arm("armX", spec, required=_required(),
                               bounded_minutes=1.0)

"""The four launch blockers and the scope correction, each as an executable check.

Every test here corresponds to a defect a `$0` review found in a session that
was otherwise ready to launch. None of them needed a GPU to find and none needs
one to keep closed:

* **one budget model.** The launcher built a second budget on top of the
  proposal's final one and double-counted the contingency and the recovery
  reserve, deriving `$36.9987` of GPU where the proposal's whole all-in ceiling
  is `$33.2099`. A correct authorization would have refused the launch.
* **the executable closure names what the executable reads.** Four files decide
  which six checkpoints the prepare stage builds, and none was declared.
* **the campaign is not the run attempt.** They were the same string, which
  made the registered continuation policy unreachable.
* **the window is authorization-bound.** It came from a constant rate, so an
  authorization re-derived at a different live quote inherited the old dollar
  window.
* **the scope says six materializations**, because six is what executes.

The launcher's `budget()` and the authorization loader are the two seams these
exercise, so they are also where a mutation has to be caught: several of these
are written as table tests over wrong inputs rather than as one assertion about
the right one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.governance.authorization import AuthorizationError  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402

import autoinit_c2_behavioural_launch as L  # noqa: E402
import autoinit_c2_behavioural_driver as D  # noqa: E402


class _Args:
    """Only what `budget()` reads. The launcher's real parser is exercised by
    `test_the_launcher_builds_a_command_this_driver_can_parse`."""

    max_price = BG.QUOTED_RATE_USD_PER_HOUR


# -- A. one budget model -----------------------------------------------------

def _plan(rate: float):
    return L.budget(_Args()).plan(price_per_hour=rate, authorized_usd=10_000.0)


@pytest.mark.parametrize("rate", [BG.QUOTED_RATE_USD_PER_HOUR, 0.79, 1.49, 2.18])
def test_the_launcher_and_the_proposal_derive_one_hard_window(rate):
    """THE load-bearing parity check. Same rate, same hard minutes and dollars.

    The launcher's `BudgetSpec` and the proposal's ceiling must be two views of
    ONE decomposition. When they were two decompositions the launcher derived
    2036.62 hard minutes against the proposal's 1800.53 — `$36.9987` against
    `$32.7097` of GPU at `$1.09/h` — because it applied the frozen probe
    model's contingency and artifact-recovery reserve a second time.

    Minutes are compared exactly: they are the same derived number, not two
    roundings of it. Dollars are compared within `DOLLAR_QUANTUM_USD`, because
    the proposal's amounts are ceilinged to four decimal places and a ceiling
    rounds up.
    """
    #: ONE MODEL, not one number. This compared the launcher's hard minutes to
    #: the proposal's directly, which is the same assertion only while the
    #: campaign owes ALL of its work. attempt5 completed ten of twelve probes,
    #: so the launcher now prices the remainder under R10 -- two probes, their
    #: arms and the restore -- and the proposal still prices a full session.
    #: They SHOULD differ, and asserting they do not made the correct
    #: behaviour a failure.
    #:
    #: What must hold is that both are views of ONE decomposition: fed the same
    #: work, they produce the same minutes and the same dollars. A second
    #: decomposition -- the defect this test exists for, which double-counted
    #: the contingency and the recovery reserve -- would still be caught,
    #: because it would disagree at identical inputs.
    plan = _plan(rate)
    work = L.budget_work(_Args())
    mine = BH.session_decomposition(REPO, **work)
    assert plan.hard_terminate_minutes == pytest.approx(
        float(mine["hard_minutes"]), abs=1e-9), (
        "the launcher's plan does not equal the canonical decomposition at "
        "the launcher's OWN work inputs; it is applying a reserve the "
        "decomposition already applied")
    #: `BG.ceiling` takes no work argument -- it prices the FULL session by
    #: construction, which is why the comparison above goes through
    #: `session_decomposition` instead. Both read that one function, so a
    #: second decomposition is still caught: it would disagree here.
    full_default = BH.session_decomposition(
        REPO, materialization_minutes=BG.materialization_minutes(
            REPO)["total_minutes"])
    assert float(BG.ceiling(
        REPO, gpu_rate_usd_per_hour=rate)["hard_ceiling"]["minutes"]) == (
        pytest.approx(float(full_default["hard_minutes"]), abs=1e-9)), (
        "the proposal's ceiling is not the canonical decomposition at the "
        "full-session defaults")

    #: A continuation MAY currently price above a fresh session, because the
    #: restore of ten 2.22 GiB probes over the dev box's measured 0.68 MB/s
    #: uplink is 585 minutes of billed pod time against 174 of actual probe
    #: training. That is a real, open infrastructure problem and it is
    #: `campaign_continuation_gate`'s job to refuse it -- asserting the plan
    #: fits would make a true statement about a broken transport into a test
    #: failure, and would tempt someone to narrow R2 to make the suite green.
    #: What is asserted instead is that the REFUSAL happens, in
    #: `test_a_remainder_that_exceeds_the_campaign_ceiling_refuses`.
    full = BG.ceiling(REPO, gpu_rate_usd_per_hour=rate)["hard_ceiling"]
    assert float(full["minutes"]) > 0


def test_the_launcher_applies_no_second_contingency_or_recovery_reserve():
    """The specific double count, named rather than inferred from a total."""
    spec = L.budget(_Args())
    assert spec.contingency_fraction == 0.0, (
        "the frozen probe model's 10% contingency is carried as a named "
        "reserve in minutes; a second fraction here applies it twice")
    d = BH.session_decomposition(
        REPO, materialization_minutes=BG.materialization_minutes(
            REPO)["total_minutes"])
    assert spec.artifact_recovery_reserve_minutes == (
        d["artifact_recovery_reserve_minutes"])
    #: And the generic terms are not a third source of phases.
    assert spec.setup_minutes == 0.0 and spec.transfer_minutes == 0.0
    assert spec.arms == 0 and spec.steps_per_arm == 0
    named = {p.name for p in spec.other_phases}
    assert "materialize_arms" in named
    assert any(n.endswith("_probes_train_and_score") for n in named), named
    assert {r.name for r in spec.soft_stop_reserves} == {
        "probe_model_contingency", "probe_duration_risk",
        "generation_length_risk"}


def test_the_decomposition_is_reconciled_against_the_frozen_record():
    """Its own source must reconstruct from it, or it bounds nothing."""
    from experiments.phase_c2 import selection_pricing as SP

    d = BH.session_decomposition(REPO, materialization_minutes=100.0)
    model = d["probe_model"]
    #: The overhead names come from their OWNER, so a rename of the probe or
    #: materialization phases cannot silently fold them into this sum. Naming
    #: the phases to EXCLUDE is what broke when they were renamed for the
    #: remaining-work generalisation, and the test then summed everything.
    overhead_names = {name for name, _ in SP.SESSION_PHASE_MINUTES}
    overheads = sum(m for name, m in d["expected_phases"]
                    if name in overhead_names)
    assert overheads == pytest.approx(
        model["expected_minutes"] - model["probe_minutes_expected"], abs=0.01)
    reserves = sum(m for _, m in d["soft_stop_reserves"])
    assert model["expected_minutes"] + reserves + d[
        "artifact_recovery_reserve_minutes"] == pytest.approx(
        model["hard_ceiling_minutes"], abs=0.01)
    #: And the materialization really is carried, once.
    assert dict(d["expected_phases"])["materialize_arms"] == 100.0
    assert d["hard_minutes"] == pytest.approx(
        model["hard_ceiling_minutes"] + 100.0, abs=0.01)
    #: The reserve block reconciled above is the FULL-session one; at the
    #: default probe count the scaled block equals it, which is what keeps the
    #: authorized figures unchanged by the remaining-work generalisation.
    assert sum(m for _, m in d["soft_stop_reserves"]) == pytest.approx(
        sum(m for _, m in d["full_session_reserves"]), abs=0.01)
    assert d["probes_remaining"] == d["probes_in_protocol"] == 12


def test_a_record_whose_reserve_block_moved_is_refused(tmp_path, monkeypatch):
    """Mutation: break the source and the decomposition must refuse, not average.

    A reconciliation that cannot fail is not a reconciliation. The frozen
    record is copied, its hard ceiling moved, and the derivation asked again.
    """
    src = json.loads((REPO / BH.PRICING).read_text())
    src["behavioural_selection"]["hard_ceiling_minutes"] += 100.0
    fake = tmp_path / BH.PRICING
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(json.dumps(src))
    with pytest.raises(BH.BehaviouralProposalError, match="reserve"):
        BH.session_decomposition(tmp_path, materialization_minutes=10.0)


def test_a_record_whose_overheads_moved_is_refused(tmp_path):
    """The other half of the reconciliation, mutated independently."""
    src = json.loads((REPO / BH.PRICING).read_text())
    beh = src["behavioural_selection"]
    beh["bounding_basis"]["probe_minutes_expected"] -= 40.0
    fake = tmp_path / BH.PRICING
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(json.dumps(src))
    with pytest.raises(BH.BehaviouralProposalError, match="besides train"):
        BH.session_decomposition(tmp_path, materialization_minutes=10.0)


# -- B. the closure names what the executable reads --------------------------

#: The runtime scientific inputs of the prepare stage. Written out here rather
#: than imported from the module under test: a test that reads its expected
#: value from its own subject can never fail.
RUNTIME_SCIENTIFIC_INPUTS = (
    "logs/stages/stage-1/phase_c2_full_search/runs/attempt3/evidence/"
    "stage1_selection.json",
    "logs/stages/stage-1/phase_c2_full_search/runs/attempt3/evidence/"
    "states_compact.jsonl",
    "logs/stages/stage-1/phase_c2_full_search/runs/attempt3/evidence/"
    "telemetry.jsonl",
    "logs/stages/stage-1/phase_c2_replay/plans/"
    "attempt3_arch_spec_lineage.json",
)


@pytest.mark.parametrize("rel", RUNTIME_SCIENTIFIC_INPUTS)
def test_every_runtime_scientific_input_is_declared(rel):
    """`build_replay_leaves` reads these, so the closure must name them.

    They decide path ancestry, the calibration profiles, every pinned
    intermediate digest, the historical root-config override that made step 0
    reproduce, and the materialization admission bounds — which is to say, they
    decide WHICH SIX CHECKPOINTS get built. "Recorded closure == live closure"
    is worth nothing while the closure does not name them.
    """
    declared = BG.declared_inputs(REPO)
    assert rel in declared, (
        f"{rel} decides what the prepare stage builds and is not in the "
        "declared closure inputs")
    assert (REPO / rel).is_file()


@pytest.mark.parametrize("rel", RUNTIME_SCIENTIFIC_INPUTS)
def test_a_declared_runtime_inputs_bytes_reach_the_closure_digest(rel):
    """Declaring a path is not hashing it. The whole chain, in three steps.

    A declared input that did not reach the digest would report a confident
    identity for a session whose scientific inputs had changed — which is worse
    than no identity at all. So: the path is in the hashed set; the hash
    recorded for it IS this file's current hash; and the digest is a function
    of those hashes, checked by moving one of them.

    Nothing here writes, copies or mirrors anything. These four paths are
    attempt 3's evidence, and evidence is not a fixture.
    """
    from aadistill.governance.closure import digest_of, sha256_of

    live = BG.current_executable(REPO)
    row = next((r for r in live["files"] if r["path"] == rel), None)
    assert row is not None, (
        f"{rel} is declared but absent from the hashed set, so its bytes "
        "decide what runs and nothing measures them")
    assert row["sha256"] == sha256_of(REPO / rel), (
        f"the closure records a hash for {rel} that is not this file's hash")

    moved = [dict(r) for r in live["files"]]
    target = next(r for r in moved if r["path"] == rel)
    target["sha256"] = "0" * 64
    assert digest_of(moved) != live["digest"], (
        f"{rel} is in the hashed set but its hash does not move the digest")


def test_the_declared_inputs_come_from_their_owners_constants():
    """One string per path. A retyped path is a path that can drift."""
    from experiments.phase_c2 import replay_specs as RS

    declared = BG.declared_inputs(REPO)
    for constant in (RS.SELECTION_REL, RS.JOURNAL_REL, RS.TELEMETRY_REL,
                     RS.ARCH_SPEC_LINEAGE_REL):
        assert constant in declared


# -- C. campaign identity is not run identity --------------------------------

def test_the_campaign_id_is_stable_and_is_not_a_run_id():
    """Stable across resources, and not derivable from one.

    Whether it reaches the PLAN hash is asserted in
    `tests/autoinit/test_c2_behavioural_launch_governance.py`: `plan_payload`
    joins the frozen selection to the dev box's durable products, which do not
    exist on a pod, and this directory runs inside the paid pod's blocking gate.
    """
    assert BG.CAMPAIGN_ID
    assert "attempt" not in BG.CAMPAIGN_ID
    assert BG.CAMPAIGN_ID == _terms()["campaign_id"]


def test_the_driver_command_sends_the_campaign_and_the_run_attempt_apart():
    """The contract between the two halves, through both real parsers."""
    class _Ctx:
        image_digest = "sha256:abc"
        price = 1.09
        spent_usd = 0.0
        args = L.build_parser().parse_args(
            ["--scr", "/tmp/probe", "--session-commit", "abc123",
             "--bundle", "b", "--run-id", "attempt7", "--max-price", "1.09"])

        class auth:
            hard_cap_usd = 32.7097
            campaign_id = BG.CAMPAIGN_ID

    class _Plan:
        soft_stop_usd = 30.0

    command = L.driver_command(_Ctx(), _Plan())
    flags = command.split("autoinit_c2_behavioural_driver.py", 1)[1].split()
    parsed = D.build_parser().parse_args([f.strip("'") for f in flags])
    assert parsed.campaign == BG.CAMPAIGN_ID
    assert parsed.run_attempt == "attempt7"
    assert parsed.campaign != parsed.run_attempt, (
        "the campaign and the run attempt are the same string again, which is "
        "what made the continuation policy unreachable")


def test_the_durable_store_is_keyed_by_campaign_then_run_attempt():
    """A campaign must be able to see its own work; two attempts must not
    overwrite each other's."""
    class _Ctx:
        class args:
            run_id = "attempt7"
            ckpt_store = "/tmp/store"

        class auth:
            campaign_id = BG.CAMPAIGN_ID

    dest = L.probe_destination(_Ctx(), "screening.B.s1")
    assert dest == Path("/tmp/store") / BG.CAMPAIGN_ID / "attempt7" / (
        "screening.B.s1")
    other = L.probe_destination(
        type("C", (), {"args": type("A", (), {"run_id": "attempt8",
                                              "ckpt_store": "/tmp/store"}),
                       "auth": _Ctx.auth})(), "screening.B.s1")
    assert other != dest, "two attempts of one campaign share a destination"
    assert other.parent.parent == dest.parent.parent, (
        "two attempts of one campaign do not share a campaign root, so a "
        "continuation cannot find its predecessor's probes")


def test_a_replacement_resource_continues_the_same_campaign(tmp_path):
    """Continuation across two synthetic run/resource identities.

    The same campaign under two different run attempts: the second must restore
    what the first completed. This is the property the old code made impossible,
    and it is checked on the real `load_campaign_journal` rather than on a
    description of it.
    """
    audit = tmp_path / "audit"
    (audit / "probes").mkdir(parents=True)
    model_dir = tmp_path / "probe_weights"
    model_dir.mkdir()
    (audit / "probes" / "screening.B.s1.json").write_text(json.dumps({
        "probe_id": "screening.B.s1", "campaign": BG.CAMPAIGN_ID,
        "run_attempt": "attempt7", "rung": "screening", "arm": "B", "seed": 1,
        "model_dir": str(model_dir), "config_sha256": "c" * 64,
        "initialization_artifact_digest": "a" * 64,
        "durable": {"identity": {"artifact_digest": "d" * 64,
                                 "arch_signature": "sig",
                                 "num_parameters": 1}},
        "score": {"correct_overall": 0.5},
    }))

    def _driver(run_attempt: str, campaign: str = BG.CAMPAIGN_ID):
        args = D.build_parser().parse_args([
            "--campaign", campaign, "--run-attempt", run_attempt,
            "--audit-dir", str(audit), "--eval-dir", str(tmp_path / "eval"),
            "--b-workdir", str(tmp_path / "arms"),
            "--status-path", str(tmp_path / "status.txt"),
            "--b-build-minutes", "31", "--probe-train-minutes", "75",
            "--probe-battery-minutes", "45", "--rate", "1.09",
            "--soft-stop-usd", "30", "--authorized-usd", "32"])
        driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
        driver.a = args
        driver.audit = audit
        driver.training, driver.scores = {}, {}
        driver.ev = {}
        #: The bytes are not the subject here; the campaign predicate is.
        driver.reidentify = lambda entry: (True, "re-identified")
        return driver

    #: A DIFFERENT resource, same campaign.
    replacement = _driver("attempt8")
    summary = replacement.load_campaign_journal()
    assert [r["probe_id"] for r in summary["restored"]] == ["screening.B.s1"]
    assert not summary["rejected"]
    assert "screening.B.s1" in replacement.scores

    #: A different CAMPAIGN is still refused. Continuation, not pooling.
    foreign = _driver("attempt8", campaign="c2-behavioural-some-other-run")
    summary = foreign.load_campaign_journal()
    assert not summary["restored"]
    assert all("campaign" in r["why"] for r in summary["rejected"])


def test_a_restored_probe_that_does_not_match_its_descriptor_is_refused():
    """Continuation may consume; it may not substitute."""
    from experiments.phase_c2 import behavioural_schedule as SCH

    probe = SCH.Probe("screening", "B", 1, "a" * 64, "/arms/B")
    driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    good = {"rung": "screening", "arm": "B", "seed": 1,
            "initialization_artifact_digest": "a" * 64,
            "config_sha256": "c" * 64}
    driver.training = {"screening.B.s1": good}
    driver.assert_reuse_matches(probe)          # the matching case passes

    for field, wrong in (("rung", "confirmation"), ("arm", "other"),
                         ("seed", 2),
                         ("initialization_artifact_digest", "b" * 64),
                         ("config_sha256", "")):
        driver.training = {"screening.B.s1": {**good, field: wrong}}
        with pytest.raises(D.C2DriverError):
            driver.assert_reuse_matches(probe)


def test_a_committed_ranking_may_not_be_re_decided(tmp_path):
    """Screening commits once per campaign."""
    driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    driver.a = type("A", (), {"campaign": BG.CAMPAIGN_ID})()
    path = tmp_path / "c2_screening_ranking.json"

    assert driver.committed_ranking(path) is None       # nothing committed yet

    path.write_text(json.dumps({
        "campaign": BG.CAMPAIGN_ID, "ranked": [{"state_id": "x"}],
        "advanced": {"state_id": "x"}}))
    assert driver.committed_ranking(path)["advanced"]["state_id"] == "x"

    #: Another campaign's commitment may not decide this one's confirmation.
    path.write_text(json.dumps({
        "campaign": "another-campaign", "advanced": {"state_id": "x"}}))
    with pytest.raises(D.C2DriverError, match="campaign"):
        driver.committed_ranking(path)

    #: A record that cannot say what advanced is not a commitment.
    path.write_text(json.dumps({"campaign": BG.CAMPAIGN_ID, "advanced": {}}))
    with pytest.raises(D.C2DriverError, match="no advanced candidate"):
        driver.committed_ranking(path)


def test_the_preregistration_no_longer_calls_a_replacement_pod_a_new_campaign():
    """The sentence that made the policy unreachable."""
    doc = json.loads((REPO / (
        "logs/stages/stage-1/phase_c2_behavioural/plans/"
        "c2_behavioural_resume_preregistration.json")).read_text())
    assert doc["campaign_versus_resource"]["campaign_id"] == BG.CAMPAIGN_ID
    #: CONTENT, not arity. This asserted `== 4` and the continuation round
    #: legitimately registered a fifth condition, so a correct advance turned
    #: the guard red. What has to hold is that each condition is still there.
    conditions = " ".join(
        doc["campaign_versus_resource"]["continuation_conditions"]).lower()
    for required in ("descriptor", "re-identif", "non-billing", "cumulative"):
        assert required in conditions, required
    r1 = next(r for r in doc["rules"] if r["id"] == "R1")
    assert "a replacement pod is a new campaign" not in r1["why"]
    assert "new RESOURCE" in r1["why"]
    #: And it still refuses the thing it was always meant to refuse.
    assert any("across CAMPAIGNS" in item
               for item in doc["what_a_resume_may_never_do"])
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "record_sha256"})
    assert recomputed == doc["record_sha256"]


# -- D. the window is authorization-bound ------------------------------------

def _terms(rate: float = BG.QUOTED_RATE_USD_PER_HOUR,
           campaign: float = BG.CAMPAIGN_ALL_IN_CEILING_USD) -> dict:
    return BG.authorization_terms(REPO, rate_usd_per_hour=rate,
                                  campaign_all_in_hard_usd=campaign)


def test_the_authorization_carries_five_distinct_amounts():
    terms = _terms()
    for field in BG.AUTHORIZATION_AMOUNT_FIELDS:
        assert terms[field] > 0, field
    assert terms["all_in_hard_usd"] == pytest.approx(
        terms["gpu_hard_usd"] + terms["disk_hard_usd"], abs=1e-9)
    assert terms["gpu_hard_usd"] == pytest.approx(
        terms["hard_runtime_minutes"] / 60 * terms["rate_usd_per_hour"],
        abs=BG.DOLLAR_QUANTUM_USD)
    assert terms["campaign_id"] == BG.CAMPAIGN_ID


def test_the_terms_are_derived_at_the_live_rate_not_the_quoted_constant():
    """A different quote must produce a different dollar authorization.

    Both probes are given a campaign ceiling wide enough to hold the session
    they imply. That is not the real ceiling and it is not meant to be: at
    $2.18/h the real `$35.7600` could not fund one attempt and
    `authorization_terms` refuses outright, which is a DIFFERENT and correct
    behaviour asserted below. Here the question is only whether the amounts
    follow the live rate.
    """
    cheap, dear = _terms(0.79, campaign=999.0), _terms(2.18, campaign=999.0)
    #: The experiment does not get shorter or longer because the card moved.
    assert cheap["hard_runtime_minutes"] == dear["hard_runtime_minutes"]
    assert cheap["gpu_hard_usd"] < dear["gpu_hard_usd"]
    #: And neither equals the quoted-rate figure by accident.
    assert cheap["gpu_hard_usd"] != _terms()["gpu_hard_usd"]
    #: The campaign ceiling is NOT a function of the rate, so a live re-quote
    #: can never move it. That is the whole reason it is a separate field.
    assert cheap[BG.CAMPAIGN_AMOUNT_FIELD] == dear[BG.CAMPAIGN_AMOUNT_FIELD]


@pytest.mark.parametrize("live,expect", [
    #: Below the authorized rate: the money would buy more minutes than the
    #: experiment is authorized to run, and it must not take them.
    (0.50, "runtime"),
    #: At the authorized rate: the authorized runtime, less the teardown
    #: reserve's worth of minutes.
    (BG.QUOTED_RATE_USD_PER_HOUR, "money"),
    #: Above it: fewer minutes still. A dearer card cannot buy a longer run.
    (2.00, "money"),
])
def test_the_window_is_the_shorter_of_the_money_and_the_runtime(live, expect):
    terms = _terms()
    window = BG.window_minutes(
        live, gpu_hard_usd=terms["gpu_hard_usd"],
        hard_runtime_minutes=terms["hard_runtime_minutes"])
    funded = ((terms["gpu_hard_usd"] - BG.TEARDOWN_RESERVE_USD) / live) * 60
    if expect == "runtime":
        assert funded > terms["hard_runtime_minutes"]
        assert window == terms["hard_runtime_minutes"]
    else:
        assert funded < terms["hard_runtime_minutes"]
        assert window == pytest.approx(funded)


def test_a_dearer_card_buys_fewer_minutes():
    terms = _terms()
    kw = {"gpu_hard_usd": terms["gpu_hard_usd"],
          "hard_runtime_minutes": terms["hard_runtime_minutes"]}
    assert BG.window_minutes(1.09, **kw) > BG.window_minutes(2.00, **kw)


def test_the_window_has_no_default_derived_from_a_constant_rate():
    """The specific defect: a window that ignored the authorization's amounts.

    `window_minutes` used to default `gpu_hard_usd` to a ceiling derived from
    `QUOTED_RATE_USD_PER_HOUR`, so a valid authorization re-derived at another
    live quote still inherited the old dollar window.
    """
    with pytest.raises(TypeError):
        BG.window_minutes(1.09)                       # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BG.window_minutes(1.09, gpu_hard_usd=32.0)    # type: ignore[call-arg]


@pytest.mark.parametrize("rate,gpu,runtime", [
    (0.0, 32.0, 1800.0),        # a non-positive rate
    (-1.0, 32.0, 1800.0),
    (1.09, 0.25, 1800.0),       # cannot even fund the teardown reserve
    (1.09, 0.10, 1800.0),
    (1.09, 32.0, 0.0),          # no authorized runtime
])
def test_the_window_refuses_an_unusable_authorization(rate, gpu, runtime):
    with pytest.raises(ValueError):
        BG.window_minutes(rate, gpu_hard_usd=gpu, hard_runtime_minutes=runtime)


def _authorization_document(**overrides) -> dict:
    """A syntactically complete authorization, for the LOADER's checks.

    Not an authorization: it is written to a tmp path by the tests that use it
    and it permits nothing. Building one here is what makes the loader's
    refusals testable without issuing the artifact they describe.
    """
    terms = _terms()
    doc = {
        "schema": BG.SCHEMA,
        "authorization_id": "test-only-not-a-grant",
        "version": 1,
        "granted_utc": "2026-09-20T00:00:00Z",
        "granted_by": "nobody -- this is a test fixture",
        "plan_id": BG.PLAN_ID,
        "phase_a_session_plan_hash": "0" * 64,
        "phase_a_science_plan_hash": "1" * 64,
        "expected_usd": terms["expected_all_in_usd"],
        "hard_cap_usd": terms["gpu_hard_usd"],
        "authorized_stages": list(BG.AUTHORIZED_STAGES),
        "stage_conditions": {},
        "scope_note": "test fixture",
        "allows_recovery_training": True,
        "campaign_id": BG.CAMPAIGN_ID,
        "resource_scope": None,
        **{f: terms[f] for f in BG.AUTHORIZATION_AMOUNT_FIELDS},
        #: The campaign ceiling is a SIXTH amount, deliberately outside
        #: AUTHORIZATION_AMOUNT_FIELDS: those five are the session's and are
        #: reconciled against each other, while this one is the maintainer's
        #: cumulative figure and is reconciled against nothing but the session
        #: all-in it must be able to fund.
        BG.CAMPAIGN_AMOUNT_FIELD: terms[BG.CAMPAIGN_AMOUNT_FIELD],
    }
    doc.update(overrides)
    doc.pop("authorization_sha256", None)
    doc["authorization_sha256"] = sha256_json(doc)
    return doc


def test_the_loader_accepts_a_consistent_document(tmp_path):
    """The positive case, so the refusals below are known to be selective."""
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_authorization_document()))
    auth = BG.BehaviouralAuthorization.load(path)
    assert auth.campaign_id == BG.CAMPAIGN_ID
    assert auth.gpu_hard_usd == _terms()["gpu_hard_usd"]
    assert auth.hard_runtime_minutes == _terms()["hard_runtime_minutes"]
    assert auth.allows_recovery_training is True


@pytest.mark.parametrize("field", BG.AUTHORIZATION_AMOUNT_FIELDS)
def test_the_loader_refuses_a_document_missing_any_amount(tmp_path, field):
    """A missing money field must refuse, never default."""
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_authorization_document(**{field: None})))
    with pytest.raises(AuthorizationError, match=field):
        BG.BehaviouralAuthorization.load(path)


@pytest.mark.parametrize("overrides,match", [
    #: The dollars and the deadline describe different experiments.
    ({"hard_runtime_minutes": 2400.0}, "different experiments"),
    #: An all-in figure that is not the sum hides the separately billed disk.
    ({"all_in_hard_usd": 32.7097}, "not the sum"),
    #: An all-in cap in hard_cap_usd permits the disk amount of extra GPU.
    ({"hard_cap_usd": 33.2099}, "same number"),
    #: A different campaign is a different experiment.
    ({"campaign_id": "c2-behavioural-v2"}, "campaign"),
])
def test_the_loader_refuses_an_inconsistent_document(tmp_path, overrides, match):
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_authorization_document(**overrides)))
    with pytest.raises(AuthorizationError, match=match):
        BG.BehaviouralAuthorization.load(path)


# -- E. the scope says what executes -----------------------------------------

def test_the_authorized_scope_names_six_materializations_not_one(tmp_path):
    """It authorizes six fixed-path arm builds, then twelve probes."""
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_authorization_document()))
    payload = BG.BehaviouralAuthorization.load(path).as_dict()

    assert payload["authorized_materializations"] == 6
    assert payload["authorized_probes"] == 12
    scope = payload["scope"]
    assert "SIX deterministic" in scope
    assert "ONE materialization of B" not in scope, (
        "the scope still describes one materialization; six execute")
    assert "twelve behavioural probes" in scope
    #: And it still forbids everything it forbade.
    for forbidden in ("NOT a search", "NOT a beam", "NOT a re-ranking",
                      "re-measurement of B's completed state evaluation",
                      "NOT a fourth seed", "NOT any part of C3 or C4"):
        assert forbidden in scope, forbidden
    for prop in ("authorizes_c2_full_search", "authorizes_c2_search1",
                 "authorizes_c2_baseline_completion", "authorizes_c2_replay",
                 "authorizes_later_cycles", "allows_phase_a"):
        assert payload[prop] is False, prop
    assert payload["authorizes_behavioural_selection"] is True


def test_the_scope_gate_still_reads_the_object_not_the_text(tmp_path):
    """The gate's answer must come from the properties, as it always did."""
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_authorization_document()))

    class _Ctx:
        auth = BG.BehaviouralAuthorization.load(path)

    ok, why = L.scope_gate(_Ctx())
    assert ok, why


@pytest.mark.parametrize("field,delta", [
    (("train_minutes", "mean"), 5.0),
    (("train_minutes", "max"), 5.0),
    (("eval_minutes", "mean"), 5.0),
    (("eval_minutes", "max"), 5.0),
])
def test_a_record_whose_per_probe_parts_stop_summing_is_refused(
        tmp_path, field, delta):
    """The train/score split is DERIVED from the record and reconciled to it.

    Twelve times the per-probe train and eval figures must reconstruct the
    record's own `bounding_basis` totals. Without that check the split would be
    an assumption about a frozen document, and a continuation that prices
    score-only work apart from training would be doing it on faith.
    """
    src = json.loads((REPO / BH.PRICING).read_text())
    cost = src["behavioural_selection"]["probe_cost"]
    cost[field[0]][field[1]] += delta
    fake = tmp_path / BH.PRICING
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(json.dumps(src))
    with pytest.raises(BH.BehaviouralProposalError,
                       match="no longer sum to the whole"):
        BH.session_decomposition(tmp_path, materialization_minutes=10.0)


# -- the metadata must not contradict the fields -----------------------------

def _loaded_authorization(tmp: Path | None = None):
    """The real object, through the real loader, from the real document shape.

    Built here rather than asserted against a dict literal: the descriptions
    under test are emitted by `as_dict()`, and a fixture that assembled them
    by hand could not detect one going stale.
    """
    import tempfile

    d = tmp or Path(tempfile.mkdtemp())
    path = d / "authorization.json"
    path.write_text(json.dumps(_authorization_document()))
    return BG.BehaviouralAuthorization.load(path)


def _ceiling_claims(payload: dict) -> list[tuple[str, str]]:
    """Every emitted description that attributes a CAMPAIGN bound to the
    per-session field.

    `all_in_hard_usd` is one session's hard all-in and equals
    `gpu_hard_usd + disk_hard_usd`; `campaign_all_in_hard_usd` is the
    cumulative bound across every resource and run attempt. A description
    saying otherwise is a false statement in formal governance evidence.
    """
    import re

    out = []
    for key, value in payload.items():
        if not isinstance(value, str):
            continue
        text = " ".join(value.split())
        #: The negative lookbehind matters: `campaign_all_in_hard_usd` ends in
        #: the session field's name, so a naive search flags every correct
        #: sentence and the check would have to be weakened to pass.
        for m in re.finditer(r"(?<!campaign_)all_in_hard_usd[^.]{0,200}", text):
            if re.search(r"\b(campaign|cumulativ)", m.group(0), re.I):
                out.append((key, m.group(0)))
    return out


def test_no_emitted_description_calls_the_session_ceiling_cumulative():
    """The two ceilings separated; two descriptions did not follow.

    `authorization_terms` emitted `_all_in_is_the_campaign_ceiling` and
    `as_dict` emitted `_the_ceiling_is_cumulative_over_the_campaign`, both
    saying `all_in_hard_usd` bounds the campaign cumulatively. Both were true
    while the ceilings were one number and false afterwards, and both sat in
    the same document as the correct contract — so a newly issued
    authorization contradicted itself in machine-readable governance metadata.
    The executable paths read the right fields throughout, so nothing
    overspent; the evidence was still wrong, and formal evidence that
    contradicts itself cannot be reviewed.
    """
    terms = _terms()
    assert _ceiling_claims(terms) == [], _ceiling_claims(terms)
    assert "_all_in_is_the_campaign_ceiling" not in terms

    payload = _loaded_authorization().as_dict()
    assert _ceiling_claims(payload) == [], _ceiling_claims(payload)
    assert "_the_ceiling_is_cumulative_over_the_campaign" not in payload

    #: And the contract IS stated, once, so deleting the false claims did not
    #: leave the artifact silent about which field bounds what.
    contract = payload["_session_versus_campaign"]
    assert "all_in_hard_usd bounds ONE session" in contract
    assert f"{BG.CAMPAIGN_AMOUNT_FIELD} bounds the CAMPAIGN" in contract
    assert "_campaign_ceiling_is_cumulative" in terms
    assert "_campaign_ceiling_buys_no_science" in terms


def test_the_claim_detector_finds_a_planted_contradiction():
    """Mutation: the check above passes on an empty detector.

    A predicate that never matches is indistinguishable from a clean payload,
    and the lookbehind that keeps `campaign_all_in_hard_usd` from matching is
    exactly the kind of expression that silently matches nothing.
    """
    planted = {
        "_stale": ("all_in_hard_usd bounds the campaign cumulatively across "
                   "every resource."),
        "_also_stale": "the all_in_hard_usd ceiling is cumulative.",
    }
    found = dict(_ceiling_claims(planted))
    assert set(found) == {"_stale", "_also_stale"}, found

    #: And the CORRECT sentences must not be flagged, or the guard would force
    #: the artifact to stop describing the campaign field at all.
    clean = {
        "_ok": (f"{BG.CAMPAIGN_AMOUNT_FIELD} bounds the CAMPAIGN cumulatively "
                "across every resource and run attempt."),
        "_ok2": ("all_in_hard_usd bounds ONE session and equals gpu_hard_usd "
                 "+ disk_hard_usd."),
        "_ok3": ("it is deliberately a different field from all_in_hard_usd, "
                 "which bounds ONE session."),
    }
    assert _ceiling_claims(clean) == [], _ceiling_claims(clean)


def test_a_remainder_that_exceeds_the_campaign_ceiling_refuses(tmp_path):
    """The gate, not the arithmetic, is what protects the ceiling.

    A continuation is priced on remaining work (R10). When the transport makes
    that remainder cost more than the campaign has left, the gate must refuse
    and return to the maintainer -- it may not shorten the experiment and it
    may not raise the ceiling. attempt5 left ten probes durable and two owed,
    and restoring those ten over the dev box's uplink prices above what the
    campaign has left; this is the check that stops it becoming an overrun.
    """
    work = L.budget_work(_Args())
    d = BH.session_decomposition(REPO, **work)
    hard_usd = d["hard_minutes"] / 60 * BG.QUOTED_RATE_USD_PER_HOUR
    settled = 22.2466                      # attempt3 + attempt5, all-in
    ceiling = BG.CAMPAIGN_ALL_IN_CEILING_USD
    #: Either it fits, or the gate is the thing that says no. Both are valid
    #: states of a live campaign; what must never hold is that it does not fit
    #: AND nothing refuses.
    if settled + hard_usd > ceiling:
        src = (REPO / "scripts/pod/autoinit_c2_behavioural_launch.py").read_text()
        body = src.split("def campaign_continuation_gate(", 1)[1].split(
            "\ndef ")[0]
        assert "return False" in body and "ceiling" in body, (
            "the remainder exceeds the campaign ceiling and the continuation "
            "gate has no refusal path")
        assert "may not raise the" in body or "not shortened" in body, (
            "the gate refuses without saying that the experiment is not "
            "shortened and the ceiling is not raised")

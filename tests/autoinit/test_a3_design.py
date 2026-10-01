"""The A3 design: its derivations, its gates, and its price.

Three things are checked here and they are different in kind.

**The gates are mutated, not merely exercised.** Every predicate the
structural comparison can refuse on is driven BOTH ways. A gate asserted only
on inputs that pass it is a gate nobody has tested; this repository has
shipped one of those before and it could not have passed for any record.

**The design's numbers are checked against their sources.** The control values
must be attempt75's, the identities the preregistration's, and the fail-fast
threshold a function of the per-seed spread C3 actually observed -- so the
threshold test feeds a DIFFERENT spread and requires the answer to move. A
derivation that ignores its input is a constant with extra steps.

**The price is checked where it differs from C3's.** The overrun factor reaches
the parent replay here and did not there, because on a step-1 session the
replay is more than half the work; the ceiling rounds up; and the book a shape
is charged to follows what the session DOES, not what its scientific claim is
called.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "autoinit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from compare_a_bsz3 import (  # noqa: E402
    _counter_comparison, _incumbent_gate, _score_comparison, _timing,
    pair_rounds, void_reasons,
)
from experiments.phase_c3 import a3_pricing as pricing  # noqa: E402
from experiments.phase_c3 import formal_pricing as FP  # noqa: E402
from experiments.phase_c3.a_bsz3 import ABsz3Error, frozen_identities  # noqa: E402

import write_a3_design as design  # noqa: E402

PLANS = REPO / "logs/stages/stage-1/phase_c3/plans"
DESIGN = PLANS / "a3_design.json"
SUPERSEDED_SPLIT = PLANS / "a_bsz3_adoption.json"
WITHDRAWN_NI = PLANS / "a_bsz3_noninferiority.json"
PRICING_RECORD = PLANS / "a3_pricing.json"
STAGE_I = REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i"


# --- 1. the structural comparison's gates, driven both ways ----------------


def _record(*, digest="aa", tokens=100, executed=110, padded=10,
            repeatable=True, forwards=4):
    return {
        "artifact_digest": digest,
        "calibration_tokens": tokens,
        "digest_repeatable_within_session": repeatable,
        "observed_execution_counters": {
            "physical_forward_invocations": forwards,
            "executed_positions": executed,
            "valid_positions": executed - padded,
            "padded_positions": padded},
    }


def test_the_masking_invariant_is_checked_and_can_fail():
    """`executed - padded == calibration_tokens`, both ways round."""
    good = _counter_comparison(_record(), _record(), None)
    assert good["masking_invariant_holds"] is True

    #: One position unaccounted for: the loop says 110 ran and 10 were
    #: padding, while the collector accumulated 99.
    bad = _counter_comparison(_record(), _record(tokens=99), None)
    assert bad["masking_invariant_holds"] is False
    assert "executed_positions - padded_positions" in bad["_masking_invariant_is"]


def test_an_absent_counter_leaves_the_invariant_unknown_not_true():
    """A missing field must not read as agreement."""
    rec = _record()
    rec["observed_execution_counters"]["padded_positions"] = None
    assert _counter_comparison(rec, rec, None)["masking_invariant_holds"] is None


def test_the_zero_cost_prediction_is_compared_and_can_disagree():
    """The $0 model predicted these counters before any pod existed.

    The comparison is only worth making because the two sides are computed by
    different code: `padding_profile` from the item lengths, the observation by
    the operator's own loop. So the disagreeing case must be reachable.
    """
    zero = {"protocols": {
        "A_bsz1": {"padding": {"n_groups": 4, "executed_positions": 110,
                               "padded_positions": 10}},
        "A_bsz3": {"padding": {"n_groups": 2, "executed_positions": 110,
                               "padded_positions": 10}}}}
    held = _counter_comparison(_record(forwards=4), _record(forwards=2), zero)
    assert held["prediction_held"] is True

    #: bsz3 was predicted to need 2 forwards and ran 3.
    missed = _counter_comparison(_record(forwards=4), _record(forwards=3), zero)
    assert missed["prediction_held"] is False
    assert missed["prediction_vs_observation"]["A_bsz3"]["forwards_agree"] is False


def test_no_prediction_is_reported_as_unavailable_not_as_agreement():
    out = _counter_comparison(_record(), _record(), None)
    assert "prediction_held" not in out
    assert "_prediction_unavailable" in out


def test_the_incumbent_gate_has_three_answers():
    """Match, mismatch, and not-asked -- and not-asked is not a pass."""
    assert _incumbent_gate("abc", "abc")["matches"] is True
    assert _incumbent_gate("abc", "def")["matches"] is False
    unchecked = _incumbent_gate("abc", None)
    assert unchecked["checked"] is False
    assert "matches" not in unchecked


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda r, c: None, 0),
        #: a masking defect
        (lambda r, c: c["execution_counters"].__setitem__(
            "masking_invariant_holds", False), 1),
        #: the two protocols averaged over different denominators
        (lambda r, c: c.__setitem__("calibration_tokens_identical", False), 1),
        #: one protocol does not reproduce its own digest
        (lambda r, c: r["A_bsz3"].__setitem__(
            "digest_repeatable_within_session", False), 1),
        #: the reference protocol is not the incumbent
        (lambda r, c: c["incumbent_digest_gate"].update(
            {"checked": True, "matches": False, "observed": "x" * 64,
             "expected": "y" * 64}), 1),
        #: everything at once
        (lambda r, c: (c.__setitem__("calibration_tokens_identical", False),
                       r["A_bsz1"].__setitem__(
                           "digest_repeatable_within_session", False)), 2),
    ],
    ids=["clean", "masking", "denominators", "repeatability", "incumbent",
         "two-at-once"])
def test_every_void_reason_is_reachable(mutate, expect):
    """A void comparison is not a result, so each way of becoming one must be
    detectable -- including when several hold together."""
    results = {"A_bsz1": _record(), "A_bsz3": _record(digest="bb")}
    comparison = {
        "calibration_tokens_identical": True,
        "execution_counters": {"masking_invariant_holds": True},
        "incumbent_digest_gate": {"checked": False},
    }
    mutate(results, comparison)
    assert len(void_reasons(results, comparison)) == expect


# --- 2. the runtime measurement protocol -----------------------------------


def _round(i, warm, scorer):
    return {"round": i, "warm_up": warm, "scorer_seconds": scorer,
            "suffix_seconds": scorer + 1.0, "peak_vram_gib": None,
            "artifact_digest": "aa"}


def test_the_warm_up_round_is_excluded_from_the_timing():
    """It is excluded by NAME, not by being first.

    The first round of each protocol pays for allocator growth, autotuning and
    kernel selection. Including it would report the arms' order as part of the
    protocol difference.
    """
    t = _timing([_round(0, True, 99.0), _round(1, False, 10.0),
                 _round(2, False, 12.0)])
    assert t["n_timed_rounds"] == 2
    assert t["warm_up_rounds_excluded"] == 1
    assert t["_fell_back_to_the_warm_up"] is False
    assert t["scorer_seconds"]["values"] == [10.0, 12.0]
    assert t["scorer_seconds"]["mean"] == 11.0


def test_a_single_round_falls_back_and_says_so():
    """Reporting nothing would be worse than reporting a warm number that
    declares itself warm."""
    t = _timing([_round(0, True, 10.0)])
    assert t["n_timed_rounds"] == 1
    assert t["_fell_back_to_the_warm_up"] is True
    assert t["warm_up_rounds_excluded"] == 0


def test_an_unpaired_round_is_dropped_rather_than_widening_one_sample():
    """Interleaving only means something while the rounds stay paired.

    A cleanup failure can stop the loop between the two protocols. The odd
    round out is precisely the one a drifting device moves, so keeping it
    would give one protocol an extra sample taken at a moment the other never
    saw.
    """
    paired, common, dropped = pair_rounds({"a": [1, 2, 3], "b": [1, 2, 3]})
    assert common == 3 and dropped == {}
    assert paired["a"] == [1, 2, 3]

    paired, common, dropped = pair_rounds({"a": [1, 2, 3], "b": [1, 2]})
    assert common == 2
    assert dropped == {"a": 1}
    assert paired["a"] == [1, 2] and paired["b"] == [1, 2]


def test_rank_correlation_separates_scores_from_selections():
    """Two protocols can differ in every score and agree on every selection.

    The counts cannot see that and this can, so identical and perturbed
    inputs must give different answers.
    """
    base = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    same = _score_comparison(base, [row[:] for row in base])
    assert same["identical"] is True
    assert same["rank_correlation"]["overall"] == pytest.approx(1.0)
    assert same["score_drift"]["abs_max"] == 0.0

    #: Order preserved, values moved: a correlation of 1.0 with real drift.
    scaled = _score_comparison(base, [[v * 1.01 for v in row] for row in base])
    assert scaled["identical"] is False
    assert scaled["rank_correlation"]["overall"] == pytest.approx(1.0)
    assert scaled["score_drift"]["abs_max"] > 0.0

    #: Order broken: the correlation must fall.
    swapped = _score_comparison(base, [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    assert swapped["rank_correlation"]["overall"] < 1.0


def test_mismatched_score_traces_are_unavailable_rather_than_zero():
    assert "_unavailable" in _score_comparison(None, [[1.0]])
    assert "_unavailable" in _score_comparison([[1.0, 2.0]], [[1.0], [2.0]])



# --- 3. A3 is ONE chain, and its numbers come from their sources -----------


def test_the_frozen_identities_are_read_and_refuse_to_be_invented(tmp_path):
    ids = frozen_identities()
    prereg = json.loads((PLANS / "c3_preregistration.json").read_text())
    assert ids["incumbent_artifact_digest"] == \
        prereg["arms"]["A_incumbent"]["artifact_digest"]
    assert ids["shared_parent_artifact_digest"] == \
        prereg["shared_parent"]["artifact_digest"]
    assert ids["recovery_seeds"] == prereg["seeds"]["recovery"]

    with pytest.raises(ABsz3Error):
        frozen_identities(tmp_path / "nope.json")
    hollow = tmp_path / "hollow.json"
    hollow.write_text(json.dumps({"seeds": {"recovery": [1]}}))
    with pytest.raises(ABsz3Error):
        frozen_identities(hollow)


def test_the_design_is_one_chain_and_not_two_steps():
    """The shape the 2026-10-02 decision asked for, asserted structurally."""
    d = json.loads(DESIGN.read_text())
    assert d["chain"] == ["initialization", "recovery_training", "evaluation",
                          "aggregation", "closeout"]
    #: No step-shaped keys survive. A design that still carried `step_1` and
    #: `step_2` alongside a `chain` would describe two workflows at once.
    assert "step_1" not in d and "step_2" not in d
    assert d["recovery"]["probes"] == 3
    assert d["recovery"]["controls_are_not_retrained"] is True
    assert d["recovery"]["seeds"] == frozen_identities()["recovery_seeds"]


def test_a_differing_digest_is_a_finding_and_not_a_stop():
    """The single most important change, and it is checked from both sides."""
    d = json.loads(DESIGN.read_text())
    diagnostics = d["initialization"]
    assert "A-bsz3 artifact digest" in diagnostics["diagnostics_collected"]
    assert "FINDING" in diagnostics["_these_are_diagnostics_not_gates"]

    #: It appears in the explicit non-stop list...
    assert any("artifact digest differing" in s for s in d["not_stops"])
    #: ...and in no integrity stop. A differing digest BETWEEN protocols is
    #: the finding; a protocol that cannot reproduce its OWN digest is the
    #: stop, and conflating the two is the error this guards.
    for stop in d["integrity_stops"]:
        assert "A-bsz3 artifact digest" not in stop["condition"]
    within = [s for s in d["integrity_stops"]
              if "WITHIN its own declared numerical protocol" in s["condition"]]
    assert len(within) == 1, "the self-reproducibility stop is missing"
    assert "NOT the same as A-bsz1 and A-bsz3 differing" in within[0]["why"]


def test_every_integrity_stop_says_repair_rather_than_interpret():
    """A genuinely invalid run is repaired, never read as a result."""
    d = json.loads(DESIGN.read_text())
    assert len(d["integrity_stops"]) >= 5
    for stop in d["integrity_stops"]:
        assert set(stop) == {"condition", "why", "action"}
        assert "repair" in stop["action"].lower(), stop


def test_the_seed_noise_band_is_derived_and_is_no_longer_a_stop():
    """Feed it a different spread and the answer must move; and it must not
    be reachable as a decision rule any more."""
    real = json.loads((STAGE_I / "c3_decision.json").read_text())
    got = design.seed_level_noise(real)
    assert got["observed_max_abs_per_seed_delta"] == \
        pytest.approx(0.022353, abs=1e-6)
    assert got["reporting_reference_only"] is True
    assert "_not_a_stop" in got and "_not_an_equivalence_margin" in got

    quiet = {"contrasts": {"p": {"delta": 0.0,
                                 "per_seed_delta": [0.001, -0.002, 0.0015]}}}
    assert design.seed_level_noise(quiet)[
        "observed_max_abs_per_seed_delta"] == pytest.approx(0.002)

    #: And the withdrawn threshold is gone from the module, not merely unused.
    assert not hasattr(design, "seed1_stop_threshold")
    assert "-0.030" not in json.dumps(json.loads(DESIGN.read_text())
                                      ["integrity_stops"])


def test_the_controls_are_attempt75s_values_and_not_restated():
    d = json.loads(DESIGN.read_text())
    probes = json.loads((STAGE_I / "c3_probe_results.json").read_text())
    source = {p["seed"]: p for p in probes["probes"]
              if p["arm"] == "A_incumbent"}
    control = d["control"]
    assert control["source_run"] == "attempt75"
    assert control["retrained"] is False
    assert len(control["probes"]) == 3
    for row in control["probes"]:
        src = source[row["seed"]]
        assert row["correct_overall"] == src["rates"]["correct_overall"]
        assert row["usable_rollout_rate"] == src["rates"]["usable_rollout_rate"]
        assert row["correct"] == src["counts"]["correct"]
        assert row["per_sample_sha256"] == src["per_sample_sha256"]


def test_the_controls_evidence_outlived_their_retired_weights():
    """The retirement removed nothing this comparison reads.

    Asserted against the retirement record itself rather than by assumption:
    the nine retired objects are all `model.safetensors`, and the fields the
    control arm consumes are scores and per-sample hashes.
    """
    retire = json.loads(
        (REPO / "logs/maintenance/inventories/"
                "archival_retirement_20261002.json").read_text())
    assert retire["schema"] == "aadistill.archival_retirement/v1"
    assert all(r["path"].endswith("/model.safetensors")
               for r in retire["retired"])
    assert retire["executed"]["objects_retired"] == len(retire["retired"]) == 9
    #: Evidence, not bytes.
    control = json.loads(DESIGN.read_text())["control"]
    consumed = set().union(*(set(r) for r in control["probes"]))
    assert "per_sample_sha256" in consumed and "correct_overall" in consumed
    assert not any("safetensors" in k or "weights" in k for k in consumed)


def test_the_full_reporting_set_is_declared():
    """The maintainer enumerated what the closeout must report."""
    d = json.loads(DESIGN.read_text())
    report = " ".join(d["aggregation_and_reporting"]["report_at_minimum"]).lower()
    for required in ("pooled and per-seed correct_overall",
                     "paired prompt-level deltas", "bootstrap interval",
                     "mcnemar", "usable_rollout_rate and all component rates",
                     "capability, domain and set breakdowns",
                     "head-map differences", "rank correlation",
                     "speedup", "peak vram", "artifact identities",
                     "consumed input hashes", "total gpu time and actual cost"):
        assert required in report, required
    assert d["aggregation_and_reporting"]["interval_is_descriptive"]


def test_it_refuses_the_formal_proof_claim():
    d = json.loads(DESIGN.read_text())
    forbidden = " ".join(d["_what_may_not_be_claimed_FROM_it"]).lower()
    assert "non-inferiority proof" in forbidden
    assert "not_a_formal_proof" in d["claim_boundary"]
    assert "DIFFERENT SESSIONS" in d["claim_boundary"]["the_cross_session_limitation"]


def test_the_identity_obligation_is_forward_only_and_does_not_stop_a3():
    """A differing artifact is recorded, not resolved, and not built around."""
    d = json.loads(DESIGN.read_text())
    sem = d["identity_semantics"]
    assert "ExecutionConfig" in sem["what_it_does_NOT_bind"]
    assert "the resulting artifact digest" in sem["what_it_does_NOT_bind"]
    assert "NOTHING PROCEDURAL" in sem["what_a_differing_digest_means_for_A3"]
    after = sem["what_it_means_afterwards"]
    assert "DISTINCT NUMERICAL MATERIALIZATION PROTOCOL" in after
    assert "RESUMABLE BEAM SEARCH" in after
    assert "does not discharge" in after
    assert "NOT BUILT" in sem["_required_shape_if_it_is_ever_needed"]
    assert "activation_importance_bsz3" in sem["_forbidden_resolution"]
    assert any("materialization-identity framework" in s
               for s in d["_what_is_forbidden"])


def test_the_superseded_designs_are_marked_and_carry_no_live_hash():
    for path, successor in ((SUPERSEDED_SPLIT, "a3_design.json"),
                            (WITHDRAWN_NI, "a3_design.json")):
        doc = json.loads(path.read_text())
        assert doc["status"] in {"SUPERSEDED", "WITHDRAWN"}
        assert "design_sha256" not in doc, (
            "a live design hash is a thing a chain binds to")
        assert "preregistration_sha256" not in doc
    split = json.loads(SUPERSEDED_SPLIT.read_text())
    assert split["superseded_by"] == "a3_design.json"
    #: The withdrawn non-inferiority design still points somewhere live.
    ni = json.loads(WITHDRAWN_NI.read_text())
    assert (PLANS / ni["superseded_by"]).is_file()
    #: And the A3 design names both of them.
    plans = {s["plan"].split("/")[-1]
             for s in json.loads(DESIGN.read_text())["supersedes"]}
    assert plans == {"a_bsz3_noninferiority.json", "a_bsz3_adoption.json"}


def test_the_committed_design_matches_what_the_writer_produces_today():
    """A committed derived record expires silently. Regenerate and compare."""
    committed = json.loads(DESIGN.read_text())
    fresh = design.build(rate_note=committed["pricing"]["_rate_note"])
    committed.pop("design_sha256")
    assert fresh == committed


def test_the_designs_self_hash_is_over_its_own_content():
    import hashlib
    committed = json.loads(DESIGN.read_text())
    claimed = committed.pop("design_sha256")
    assert hashlib.sha256(json.dumps(
        committed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == claimed


# --- 4. the price of the whole chain, and every limit that binds it --------


def test_the_overrun_factor_reaches_the_parent_replay():
    """C3 scaled training and evaluation only, which was right at nine probes
    and is not on a chain whose replay is 22 of 320 expected minutes."""
    hard = pricing.component_minutes(worst=True)
    exp = pricing.component_minutes(worst=False)
    assert hard["parent_replay"] == pytest.approx(
        exp["parent_replay"] * pricing.OVERRUN_FACTOR)
    assert hard["setup"] == pricing.SETUP_WORST_CASE_MINUTES
    assert hard["pre_provider_gates"] == exp["pre_provider_gates"]


def test_the_hard_ceiling_rounds_up_and_exceeds_the_expected():
    p = pricing.price_a3(1.09)
    exact = p.hard_minutes / 60.0 * p.billed_rate_usd_per_hour
    assert p.hard_usd >= exact and p.hard_usd - exact < 1e-4
    assert p.hard_usd > p.expected_usd


def test_the_chain_prices_three_probes_and_three_evaluations():
    """One shape, and it contains the whole experiment."""
    exp = pricing.component_minutes(worst=False)
    per_probe = 555.85 / 9
    assert exp["recovery_probes"] == pytest.approx(3 * per_probe, abs=0.02)
    assert exp["evaluations"] == pytest.approx(3 * 241.63 / 9, abs=0.02)
    for required in ("setup", "parent_replay", "initialization_rounds",
                     "aggregation", "collect_and_teardown"):
        assert exp[required] > 0, required


def test_storage_is_derived_from_the_recipe_and_two_resources_stay_apart():
    st = pricing.storage_requirement(3)
    #: The trained probe is NOT the size of the leaf it started from -- that
    #: error provisioned a pod at half the real figure and it ran out of disk.
    assert st["footprints_gib"]["trained_probe"] > \
        st["footprints_gib"]["initialization_leaf_bf16"]
    assert st["training_dtypes"]["save_dtype"] == "float32"
    #: Container residency and the durable requirement are different resources.
    assert st["durable_requirement"]["gib"] < st["container_residency"]["peak_gib"]
    assert st["provision"]["headroom_gib"] > 0
    #: And the provision is derived, not C3's 120 GB carried forward.
    assert st["provision"]["container_disk_gb"] < 120


def test_a_recipe_with_an_unknown_dtype_refuses_to_be_priced(tmp_path):
    """A storage bound may not guess a dtype: the ratio between two dtypes is
    exactly the factor by which the bound would be wrong."""
    bad = tmp_path / "configs/stage3/e1"
    bad.mkdir(parents=True)
    (bad / "e1_r0860k_sa_pca.json").write_text(
        json.dumps({"dtype": "float9", "optim": {"betas": [0.9, 0.95]}}))
    with pytest.raises(pricing.A3PricingError, match="not a per-parameter"):
        pricing.training_dtypes(tmp_path)
    (bad / "e1_r0860k_sa_pca.json").write_text(json.dumps({"dtype": "float32"}))
    with pytest.raises(pricing.A3PricingError, match="optim.betas"):
        pricing.training_dtypes(tmp_path)


def test_a3_is_booked_to_the_formal_allowance():
    """It trains probes and consumes the frozen confirmation battery, and the
    package says the engineering allowance authorizes neither."""
    assert pricing.BOOK == "formal_allowance"
    terms = json.loads(
        (REPO / "configs/experiments/phase_c1/authorization.json").read_text())
    forbidden = " ".join(terms["execution_package"][
        "engineering_allowance_does_not_authorize"]).lower()
    assert "confirmation battery" in forbidden
    assert "complete formal probes" in forbidden


def test_all_four_limits_are_checked_including_the_package_total():
    """The gap this round repaired. Each limit must be able to refuse alone."""
    env = {"per_session_envelope_usd": 30.0, "project_cap_usd": 400.0,
           "cumulative_spend_usd": 0.0, "formal_remaining_usd": 999.0,
           "engineering_remaining_usd": 999.0, "package_remaining_usd": 999.0}
    assert pricing.assess(1.09, env)["FUNDABLE"]

    for key, value, expected in (
            ("per_session_envelope_usd", 1.0, "per_session_envelope"),
            ("cumulative_spend_usd", 399.9, "project_cap"),
            ("formal_remaining_usd", 0.5, "formal_allowance"),
            ("package_remaining_usd", 0.5, "package_total")):
        f = pricing.assess(1.09, {**env, key: value})
        assert f["FUNDABLE"] is False, key
        assert expected in f["shortfalls_usd"], (key, f["shortfalls_usd"])

    assert set(pricing.assess(1.09, env)["conditions"]) == {
        "derived_ceiling_within_session_envelope",
        "cumulative_plus_derived_within_project_cap",
        "booked_allowance_covers_derived",
        "package_total_covers_derived"}


def test_a_missing_envelope_refuses_rather_than_passing_silently():
    env = {"per_session_envelope_usd": 30.0, "project_cap_usd": 400.0,
           "cumulative_spend_usd": 0.0, "formal_remaining_usd": 999.0,
           "engineering_remaining_usd": 999.0, "package_remaining_usd": 999.0}
    for drop in FP.REQUIRED_ENVELOPE_KEYS:
        with pytest.raises(FP.C3PricingError, match="missing"):
            pricing.assess(1.09, {k: v for k, v in env.items() if k != drop})


def test_the_committed_pricing_record_says_the_chain_is_funded():
    doc = json.loads(PRICING_RECORD.read_text())
    assert doc["book"] == "formal_allowance"
    assert doc["FUNDABLE"] is True, (
        "the amendment was sized to make this true; a record saying otherwise "
        "has gone stale")
    assert doc["shortfalls_usd"] == {}
    assert len(doc["_every_applicable_limit_is_checked"]) == 4


def test_the_amendment_funds_the_chain_plus_one_restart_and_no_more():
    """Minimality, asserted rather than asserted-about."""
    import subprocess
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/consolidate/derive_budget.py"),
         "--json"], capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr[-400:]
    b = json.loads(out.stdout)
    hard = pricing.price_a3(1.09).hard_usd
    formal_left = b["formal"]["remaining_usd"]
    assert formal_left >= hard, "the chain is not funded"
    #: One pre-science restart, and not two: the remainder after a full-ceiling
    #: run must not fund another full-ceiling run.
    assert formal_left < 2 * hard, (
        f"formal remaining {formal_left} funds two full attempts; the "
        "amendment was supposed to be minimal")
    #: The identity the books are derived under still holds.
    assert (b["formal"]["allowance_usd"] + b["engineering"]["allowance_usd"]
            == pytest.approx(b["package"]["allowance_usd"]))

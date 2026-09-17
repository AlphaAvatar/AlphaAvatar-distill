"""The Phase-C2 full joint re-search: space derived, cost measured, price honest.

Zero cost, CPU only. Four things are being tested, and they are different:

1. **The space is DERIVED, not stated.** The maintainer decision that asked for
   this experiment also said not to hardcode its size, because a remembered
   number is how a search gets configured for a space nobody enumerated. So the
   sizes are recomputed here from the registry, and the modules are checked for
   not containing them as literals.

2. **The cost table is pooled from every committed run, per cell.** Taking the
   max across Phase-B attempt 5 and C2 attempt 4 is the only honest ceiling, and
   one cell moved the *wrong* way between them.

3. **The extraction is behaviour-preserving.** `search_cost_model` was pulled
   out of the Search-1 module; Search-1's own numbers must be unchanged.

4. **The price refuses rather than shrinking.** The complete chain does not fit
   the remaining headroom, and the plan documents have to say so with arithmetic
   a reader can recompute.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/consolidate"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))

from aadistill.infrastructure.budget import BudgetError  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

from experiments import search_cost_model as CM  # noqa: E402
from experiments.phase_c2 import full_search_space as FS  # noqa: E402
from experiments.phase_c2 import search_space as SS  # noqa: E402
from experiments.phase_c2 import selection_pricing as SP  # noqa: E402

PROTOCOL = REPO / "logs/stages/stage-1/phase_c2/plans/phase_c2_full_search_protocol.json"
PRICING = REPO / "logs/stages/stage-1/phase_c2/plans/phase_c2_full_search_pricing.json"


@pytest.fixture(scope="module")
def registered():
    FS.register_c2_operators()
    return True


@pytest.fixture(scope="module")
def space(registered):
    return FS.full_joint_space()


# --- 1. the space is derived ------------------------------------------------


def test_the_space_is_enumerated_from_the_registry_not_stated(space):
    """Every size claim comes from walking the real registry."""
    report = FS.size_report()
    full = report["full_joint"]

    #: Recomputed independently of `decomposition`, by counting the walk.
    leaves = list(CM.walk_leaves(space))
    assert full["total_leaves"] == len(leaves)

    #: And the decomposed subspace is the longest-path bucket, not the total.
    longest = max(len(steps) for steps, _ in leaves)
    assert full["decomposed_operator_count"] == longest
    assert full["decomposed_leaves"] == sum(
        1 for steps, _ in leaves if len(steps) == longest)
    assert full["decomposed_leaves"] < full["total_leaves"], (
        "a composite operator reaches the target in one step, so the "
        "decomposed count must be strictly smaller than the total. If they are "
        "equal the two claims have been collapsed into one.")


def test_the_decomposed_count_is_the_product_the_registry_implies(space):
    """The enumeration and the arithmetic must agree, or one of them is wrong.

    Not a substitute for the walk — the walk is authoritative because it handles
    composite operators and applicability. This checks the walk against the
    factorisation a reader would do by hand, so a silent change in either is
    visible.
    """
    import math

    options = {}
    for impl, profiles in space.options(frozenset()):
        options.setdefault(impl.kind, 0)
        options[impl.kind] += len(profiles)
    #: Only the kinds that participate in a decomposed path: the single-step
    #: composite is its own bucket and is excluded from the product.
    decomposed_kinds = {k: n for k, n in options.items()
                        if k != "COMPOSITE_STAGE1"}
    product = math.factorial(len(decomposed_kinds))
    for count in decomposed_kinds.values():
        product *= count
    assert FS.size_report()["full_joint"]["decomposed_leaves"] == product


def test_the_growth_over_phase_b_is_the_calibration_branch(registered):
    """ATTENTION now consumes calibration; that is the whole difference."""
    from aadistill.initialization.calibration.profiles import consumes_calibration
    from aadistill.initialization.operators.base import get_implementation

    promoted = get_implementation(FS.PROMOTED_ATTENTION)
    replaced = get_implementation("attention.weight_proxy_v0")
    assert consumes_calibration(promoted)
    assert not consumes_calibration(replaced)

    report = FS.size_report()
    full = report["full_joint"]["decomposed_leaves"]
    reference = report["phase_b_reference"]["decomposed_leaves"]
    #: One extra branching factor on exactly one kind, so the decomposed space
    #: grows by the number of active profiles.
    assert full == reference * len(FS.PROFILE_IDS)


def test_no_module_hardcodes_the_space_size(space):
    """The sizes must not appear as literals in the code that derives them."""
    report = FS.size_report()
    forbidden = {str(report["full_joint"]["total_leaves"]),
                 str(report["full_joint"]["decomposed_leaves"]),
                 str(report["if_the_exclusion_were_not_taken"]["total_leaves"])}
    for module in (FS, CM):
        text = Path(module.__file__).read_text()
        #: Prose may discuss a number; code may not contain it as a literal. So
        #: the check is on code lines only, comments and docstrings excluded by
        #: parsing rather than by guessing.
        import ast
        tree = ast.parse(text)
        literals = {str(node.value) for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, int)}
        leaked = forbidden & literals
        assert not leaked, (
            f"{Path(module.__file__).name} contains the derived space size "
            f"{leaked} as an integer literal; it must be enumerated")


# --- 2. the cost table -----------------------------------------------------


def test_the_cost_table_pools_both_committed_runs():
    """Per cell, the maximum over every committed search on this hardware."""
    derived = FS.derive_cost_table(REPO)
    assert set(derived["sources"]) == {"phase_b_attempt5", "phase_c2_attempt4"}

    #: Recompute one cell the hard way, from both files, and require the max.
    import collections
    import statistics
    pooled = collections.defaultdict(list)
    for name, telemetry, result in FS.TELEMETRY_SOURCES:
        rows = [json.loads(line) for line
                in (REPO / telemetry).read_text().splitlines() if line.strip()]
        if result:
            levels = json.loads((REPO / result).read_text())["levels"]
            level_of = {s: L["level"] for L in levels for s in L["generated"]}
            root_of = lambda r: level_of.get(r["state_id"]) == 0  # noqa: E731
        else:
            sids = {r["state_id"] for r in rows}
            roots = {r["parent_id"] for r in rows if r["parent_id"] not in sids}
            root_of = lambda r: r["parent_id"] in roots  # noqa: E731
        for row in rows:
            minutes = (row["operator_seconds"] + row["parent_load_seconds"]
                       + sum(FS._phase_seconds(row, k)
                             for k in FS.EXPANSION_PHASES)) / 60
            pooled[(row["impl_id"],
                    "root" if root_of(row) else "deeper")].append(minutes)
    for (impl, where), values in pooled.items():
        assert derived["minutes"][impl][f"{where}_max"] == pytest.approx(
            max(values), abs=0.01)
        assert derived["minutes"][impl][f"{where}_mean"] == pytest.approx(
            statistics.mean(values), abs=0.01)
        assert derived["observations"][impl][where] == len(values)


def test_the_depth_cell_that_got_more_expensive_is_carried(registered):
    """C2 attempt 4 observed a deeper DEPTH expansion above Phase B's max.

    This is the one cell where pooling makes the ceiling HIGHER, and it is the
    figure the whole price turns on. If pooling ever silently preferred the
    older run, the search would be priced below what it has been seen to cost.

    Measured on the POOLED table, before the measured-optimization refresh.
    That refresh legitimately takes the cell BELOW Phase B's historical max --
    the executable is faster than the one Phase B ran, measured rather than
    argued -- so reading the refreshed cell here would test the refresh and
    stop testing the pooling. Both are checked, separately.
    """
    impl = "depth.causal_kl_greedy_v1"
    pooled = FS.derive_cost_table(REPO)["minutes"][impl]
    phase_b_only = SS.MEASURED_MINUTES[impl]
    assert pooled["deeper_max"] > phase_b_only["deeper_max"], (
        "pooling preferred the older run's cheaper observation")

    #: And the drop below it is the refresh, at the factor the record states --
    #: not an observation quietly going missing from the pool.
    refreshed = FS.cost_model(REPO).minutes[impl]
    record = json.loads(
        (REPO / FS.MEASURED_OPTIMIZATION).read_text())["cells"][impl]
    factor = record["refreshed_total_minutes"] / record["observed_total_minutes"]
    assert refreshed["deeper_max"] == pytest.approx(
        pooled["deeper_max"] * factor, abs=0.01)
    assert refreshed["deeper_max"] < phase_b_only["deeper_max"], (
        "the refreshed DEPTH cell is expected to sit below Phase B's "
        "historical max, because the forward-KL-only path was measured at "
        f"{1 / factor:.2f}x on a real L40S. If this ever reverses, the "
        "refresh stopped applying and the price silently rose.")


def test_the_attention_proxy_is_retired_and_was_conservative(registered):
    """The operator has now run inside a search, so nothing is proxied."""
    cost = FS.cost_model(REPO)
    assert cost.unmeasured == (), (
        "the full joint space must have no proxied rows: every implementation "
        "in it has run inside a real search")
    measured = cost.minutes_for(FS.PROMOTED_ATTENTION, root=True)
    proxied = SS.minutes_for(FS.PROMOTED_ATTENTION, root=True)
    assert measured < proxied, (
        "Search-1's proxy should prove to have been conservative; if the "
        "measurement came in above it, the margin was in the unsafe direction "
        "and every Search-1 cost claim needs revisiting")


def test_the_cost_model_refuses_what_it_has_not_measured():
    """An unpriceable expansion must raise, never default to a number."""
    cost = CM.CostModel(minutes={"a": {"root_max": 1.0}})
    assert cost.minutes_for("a", root=True) == 1.0
    with pytest.raises(CM.CostModelError, match="no measured cost"):
        cost.minutes_for("b", root=True)
    with pytest.raises(CM.CostModelError, match="no 'deeper_max' observation"):
        cost.minutes_for("a", root=False)


def test_the_expensive_operator_is_derived_not_named(space, registered):
    """`trajectory` must not need to be told which operator is costly."""
    cost = FS.cost_model(REPO)
    assert cost.most_expensive(space.allowed_impls) == "depth.causal_kl_greedy_v1"
    #: And a cost model where something else dominates nominates that instead,
    #: which is what makes the machinery family-neutral.
    flipped = CM.CostModel(minutes={
        "depth.causal_kl_greedy_v1": {"root_max": 1.0, "deeper_max": 1.0},
        "ffn.activation_importance_v0": {"root_max": 99.0, "deeper_max": 99.0}})
    assert flipped.most_expensive(
        ("depth.causal_kl_greedy_v1", "ffn.activation_importance_v0")
    ) == "ffn.activation_importance_v0"


# --- 3. the extraction changed nothing for Search-1 -------------------------


def test_search_1_numbers_are_unchanged_by_the_extraction(registered):
    """The Search-1 plan document's figures still reproduce.

    `search_cost_model` was extracted from this module. The Search-1 space, its
    bound and its replay back-test are the regression that proves the move was
    mechanical, and the committed plan document is the independent statement of
    what they were.
    """
    back = SS.replay_phase_b()
    assert back["every_level_exact"], back["levels"]
    assert back["predicted_minutes"] == pytest.approx(
        back["observed_minutes"], rel=0.01)

    #: And the restricted space is still the restricted space.
    search1 = SS.c2_search1_space()
    assert search1.impl_profiles is not None
    assert set(search1.allowed_impls) == set(SS.C2_ALLOWED_IMPLS)


def test_the_two_spaces_are_different_experiments(registered, space):
    """Search-1 is frozen; this is its successor, not a rewrite of it."""
    search1 = SS.c2_search1_space()
    assert space.impl_profiles is None, (
        "the joint space must pin no calibration; that is the restriction "
        "being reopened")
    assert len(space.allowed_impls) > len(search1.allowed_impls)
    #: The frozen artifacts the protocol promises not to touch must all exist.
    protocol = json.loads(PROTOCOL.read_text())
    for rel in protocol["frozen_and_untouchable"]["artifacts"]:
        assert (REPO / rel).is_file(), rel


# --- 4. the exclusion is a recorded scientific claim ------------------------


def test_the_only_exclusion_cites_a_completed_verdict(registered):
    assert set(FS.EXCLUSIONS) == {"attention.weight_proxy_v0"}
    why = FS.EXCLUSIONS["attention.weight_proxy_v0"]
    assert "C1" in why and "GO" in why
    #: Cheap alternatives are IN. Excluding one on cost grounds is the failure
    #: this space exists to avoid.
    for cheap in ("depth.positional_v0", "composite.stage1_sandwich_v0"):
        assert cheap in FS.full_joint_space().allowed_impls, cheap


def test_the_cost_of_not_excluding_is_derived(registered):
    """The exclusion's price is a number, not a claim in prose."""
    report = FS.size_report()
    assert (report["if_the_exclusion_were_not_taken"]["total_leaves"]
            > report["full_joint"]["total_leaves"])


# --- 5. the price refuses instead of shrinking ------------------------------


def test_the_headroom_verdict_matches_what_plan_session_actually_does(
        space, registered):
    """Whatever the record claims about fitting, `plan_session` must agree.

    This asserted a REFUSAL while the cap was $320, with a docstring saying that
    if headroom ever rose enough for it to pass, the records were stale. The cap
    rose to $370 on 2026-09-17 and it did. So the test now checks the INVARIANT
    rather than one side of it: the pricing record's derived verdict and the
    budget module's behaviour cannot disagree.
    """
    from derive_budget import derive

    remaining = derive(REPO)["project"]["remaining_usd"]
    recorded_fits = "FITS" in json.loads(PRICING.read_text())["funding"]["status"]
    try:
        FS.price(space, price_per_hour=FS.PRICE_PER_HOUR_LAST_QUOTED,
                 authorized_usd=remaining, beam_width=6)
        actually_fits = True
    except BudgetError:
        actually_fits = False
    assert actually_fits is recorded_fits, (
        f"the pricing record says fits={recorded_fits} and plan_session says "
        f"{actually_fits}; one of them is stale")
    #: And fitting is never permission — the record must carry that either way.
    funding = json.loads(PRICING.read_text())["funding"]
    assert "NOT AUTHORIZED" in funding["status"] or not actually_fits


def test_the_funding_requirement_is_the_standing_design_not_the_cheapest():
    """The funding block must price the STANDING beam width, not the cheapest.

    A narrower beam explores less of the same space and can return a different
    front, so quoting its chain as the funding requirement would fund a
    different experiment from the one the protocol proposes. Beam 2/3/4 stay
    documented as scientific alternatives; they must not be the requirement.
    """
    from aadistill.initialization.planning.ranking import SCHEDULE_V1

    doc = json.loads(PRICING.read_text())
    blocker = doc["funding"]
    combined = doc["combined"]
    selection = doc["behavioural_selection"]

    assert blocker["standing_beam_width"] == SCHEDULE_V1.width
    standing = next(r for r in doc["search"]["widths"]
                    if r["beam_width"] == SCHEDULE_V1.width)
    #: The GPU-only chain, which is what the cost model derives and what this
    #: test used to compare against. Still recorded, because it is a real
    #: intermediate -- but it is no longer the funding requirement.
    gpu_only = round(standing["hard_ceiling_usd"]
                     + selection["hard_ceiling_usd"], 4)
    assert combined["hard_ceiling_usd"] == gpu_only
    assert blocker["complete_chain_gpu_only_hard_usd"] == gpu_only

    #: THE FUNDING FIGURE IS THE TOTAL. The provider bills Container Disk
    #: separately from the GPU and both sessions provision it, so a requirement
    #: quoted at the GPU ceiling would fund a chain the provider then exceeds.
    #: This test asserted the GPU-only sum until 2026-09-17 and was correct for
    #: a world in which storage was free; it is not that world.
    provider = doc["provider_cost"]
    total = round(provider["search"]["total_hard_ceiling_usd"]
                  + selection["hard_ceiling_usd"]
                  + provider["behavioural_selection_storage_upper_bound_usd"], 4)
    assert blocker["complete_chain_hard_usd"] == total
    assert combined["chain_total_hard_ceiling_usd"] == total
    assert total > gpu_only, (
        "the total does not exceed the GPU-only chain, so the storage term is "
        "inert")

    #: And it is NOT the cheapest chain, which is the mistake being guarded.
    cheapest = min(doc["search"]["widths"],
                   key=lambda r: r["hard_ceiling_usd"])
    if cheapest["beam_width"] != SCHEDULE_V1.width:
        assert gpu_only > round(cheapest["hard_ceiling_usd"]
                                + selection["hard_ceiling_usd"], 4)
        assert not any("cheapest" in key for key in blocker), (
            "the blocker still frames a cheapest-width chain as the "
            "requirement")

    #: The arithmetic a reviewer would redo.
    remaining = doc["budget_position"]["remaining_usd"]
    #: Against the TOTAL, which is what the shortfall now measures. Measuring
    #: it against a GPU-only ceiling understated the requirement by the storage
    #: bill, and nothing in the document disagreed.
    assert blocker["shortfall_on_hard_ceilings_usd"] == pytest.approx(
        round(total - remaining, 4), abs=1e-4)
    assert blocker["minimum_cumulative_cap_usd"] == pytest.approx(
        round(doc["budget_position"]["cumulative_spend_usd"] + total, 4),
        abs=1e-4)
    #: And the STATUS must follow the sign rather than be asserted beside it.
    #: This block once read "INSUFFICIENT PROJECT HEADROOM" while printing a
    #: negative shortfall, the moment the cap rose.
    fits = blocker["shortfall_on_hard_ceilings_usd"] <= 0
    assert ("FITS" in blocker["status"]) is fits, (
        f"status {blocker['status']!r} contradicts a shortfall of "
        f"{blocker['shortfall_on_hard_ceilings_usd']}")
    assert blocker["headroom_after_the_chain_usd"] == pytest.approx(
        -blocker["shortfall_on_hard_ceilings_usd"], abs=1e-4)
    #: Fitting is never permission, and the record must say so.
    assert "not a spend authorization" in blocker["fitting_is_not_permission"]
    assert "not transferable" in blocker["fitting_is_not_permission"]
    #: The rate is planning evidence and must be re-quoted.
    assert "re-quoted live" in blocker[
        "the_price_basis_is_planning_evidence_only"]
    assert "NOT narrowed" in blocker["if_the_rate_rises"]

    #: Narrower widths are present, and labelled as alternatives rather than
    #: offered as cost options.
    alternatives = {r["beam_width"]
                    for r in combined["scientific_alternatives_not_cost_options"]}
    assert alternatives and SCHEDULE_V1.width not in alternatives
    assert "different experiment" in blocker[
        "the_narrower_beams_are_not_the_requirement"]
    #: The rounding comparison is derived, not recounted. It is about the two
    #: GPU ceilings specifically -- it demonstrates how much a display-rounded
    #: reading of THOSE understates them -- so it stays on the GPU-only chain
    #: rather than following the total.
    rounding = blocker["_rounding"]
    assert rounding["chain_from_4dp_ceilings_usd"] == gpu_only
    assert rounding["understatement_usd"] == pytest.approx(
        gpu_only - rounding["chain_from_2dp_display_ceilings_usd"], abs=1e-6)

    #: Every width's fit flag and its refusal text must agree with each other:
    #: a width that does not fit owes the refusal, and one that fits must not
    #: carry one. Requiring at least one refusal was right while the cap was
    #: $320 and became wrong when it rose — a test that assumes a shortfall is a
    #: test that expires.
    for row in doc["search"]["widths"]:
        if row["fits_remaining_headroom"]:
            assert row["refusal"] is None, row["beam_width"]
        else:
            assert row["refusal"], row["beam_width"]


def test_multi_session_continuation_is_not_claimed_as_a_capability():
    """Content-derived ids are an identity, not the bytes.

    The frozen Search-1 plan records that the multi-gigabyte search workdir
    cannot be relayed for resume, so a fresh provider resource must re-derive
    lost state. This round implemented and validated no durable cross-session
    mechanism, and no pricing option may assume one exists.
    """
    doc = json.loads(PROTOCOL.read_text())
    block = doc["execution_capabilities_this_round_did_not_build"][
        "multi_session_continuation"]
    assert block["implemented_this_round"] is False
    assert block["validated_this_round"] is False
    assert "NOT A CURRENT CAPABILITY" in block["status"]

    #: The claim is sourced to the frozen document that actually says it.
    source = (REPO / block["source"]).read_text()
    assert "cannot be relayed for resume" in source

    #: And nothing in the pricing plans around a continuation. Scanned over the
    #: WHOLE funding block rather than one list: the options list is gone now
    #: that the maintainer has decided, and a guard keyed on a field that no
    #: longer exists would pass vacuously.
    pricing = json.loads(PRICING.read_text())
    funding = json.dumps(pricing["funding"]).lower()
    for forbidden in ("continuation", "resume", "multi-session"):
        assert forbidden not in funding, (
            f"the funding block plans around {forbidden!r}, which is not a "
            "current capability")


def test_the_budget_position_is_derived_not_restated():
    doc = json.loads(PRICING.read_text())
    from derive_budget import derive

    project = derive(REPO)["project"]
    assert doc["budget_position"]["cumulative_spend_usd"] == project[
        "cumulative_spend_usd"]
    assert doc["budget_position"]["remaining_usd"] == project["remaining_usd"]


# --- 6. the behavioural selection stage ------------------------------------


def test_the_probe_bound_is_the_observed_maximum_not_the_mean():
    """A mean scaled up is not a bound, and this module once did that.

    C1 attempt 18 emitted a per-probe marker per train and per score, so six
    observed durations of each exist. The expected path may use their means; the
    CEILING must rest on their maxima plus a named reserve.
    """
    cost = SP.measured_probe_cost(REPO)
    log = (REPO / SP.MARKER_SOURCE).read_text()
    evidence = json.loads((REPO / SP.MEASURED_SOURCE).read_text())["stages"]

    trained = SP._markers(log, SP.TRAIN_MARKER)
    scored = SP._markers(log, SP.SCORE_MARKER)
    assert len(trained) == len(scored) == int(evidence["G"]["probes_trained"])

    train = SP._consecutive_durations(
        trained, SP._utc(evidence["F"]["finished_utc"]))
    evaluate = SP._consecutive_durations(scored, trained[-1][0])
    assert cost.train_minutes_max == pytest.approx(max(train), abs=1e-6)
    assert cost.eval_minutes_max == pytest.approx(max(evaluate), abs=1e-6)

    #: The bound must strictly exceed the expected, or it is the mean wearing a
    #: ceiling's name.
    assert cost.bounding_minutes > cost.expected_minutes
    assert cost.generation_length_reserve_minutes > 0


def test_the_ceiling_rests_on_the_bound_and_names_its_reserves():
    """Every minute above the expected path is a named, derivable reserve."""
    schedule = json.loads(PROTOCOL.read_text())[
        "behavioural_selection"]["schedule"]
    doc = json.loads(PRICING.read_text())["behavioural_selection"]
    cost = SP.measured_probe_cost(REPO)
    n = schedule["total_probes"]

    basis = doc["bounding_basis"]
    assert basis["probe_minutes_expected"] == pytest.approx(
        round(n * cost.expected_minutes, 2), abs=0.02)
    assert basis["probe_minutes_observed_max"] == pytest.approx(
        round(n * cost.bounding_minutes, 2), abs=0.02)
    assert basis["generation_length_reserve_minutes"] == pytest.approx(
        round(n * cost.generation_length_reserve_minutes, 2), abs=0.02)
    #: And the ceiling is above the observed-max basis, because the reserves and
    #: the contingency sit on top of it.
    assert doc["hard_ceiling_minutes"] > basis["probe_minutes_observed_max"]
    assert doc["hard_ceiling_usd"] > doc["expected_usd"]


def test_no_probe_reuse_is_assumed():
    """The ruling is read. Assuming reuse would underfund the stage."""
    ruling = SP.reuse_is_admissible(REPO)
    assert ruling["admissible"] is False
    assert ruling["n_admitted"] == 0
    committed = json.loads((REPO / SP.REUSE_RULING).read_text())
    assert committed["reuse_verified"] is False


# --- 6b. the behavioural design is PHASE-C, not Phase-B --------------------


def test_the_behavioural_design_is_phase_c_and_not_phase_b():
    """C0 retired the Phase-A/B behavioural design; C2 must not revert to it."""
    doc = json.loads(PROTOCOL.read_text())["behavioural_selection"]
    frozen = doc["frozen_science"]
    c0 = json.loads((REPO / frozen["sources"]["c0_protocol"]).read_text())
    c1 = json.loads((REPO / frozen["sources"]["c1_execution"]).read_text())

    #: The Phase-C battery and scoring contract, by identity.
    assert frozen["battery"]["asset_id"] == c1["battery"]["asset_id"]
    assert frozen["battery"]["content_sha256"] == c1["battery"]["content_sha256"]
    assert frozen["battery"]["n_scorable_prompts"] == 850
    assert frozen["scoring_contract"]["digest"] == c1["scoring_contract"]["digest"]

    #: The Phase-C statistics, read from C0.
    assert frozen["effect_sizes"]["sesoi"] == c0["effect_sizes"]["sesoi"] == 0.01
    assert frozen["decision_rule"] == c0["decision_rule"]
    assert frozen["decision_rule"]["no_forced_winner"] is True
    assert frozen["primary_inference"]["method"] == "stratified cluster bootstrap"
    assert frozen["primary_inference"]["seeds_are_not_resampled"] is True
    assert doc["no_forced_winner"] is True

    #: And the Phase-B design is explicitly refused, not merely absent.
    refused = frozen["what_is_not_reused"]
    assert "0.011695" in refused["phase_b_equivalence_interval"]
    assert "not used" in refused["successive_halving"].lower()
    #: The interval may appear ONLY inside the sentence that refuses it. A blanket
    #: substring check over the whole block would fail on its own refusal, which
    #: is the wrong scope: what must not happen is the interval being used as a
    #: BOUNDARY anywhere else.
    elsewhere = {k: v for k, v in frozen.items() if k != "what_is_not_reused"}
    assert "0.011695296982299022" not in json.dumps(elsewhere), (
        "the Phase-B equivalence interval is being used as a boundary again")
    assert "0.011695296982299022" in json.dumps(
        frozen["what_is_not_reused"]), (
        "the refusal must name the interval it refuses, or a reader cannot "
        "tell which design was retired")
    for historical in ("20260726", "20260801", "20260813"):
        assert historical not in json.dumps(doc["seeds"]["all"]), historical


def test_the_c2_seeds_are_fresh_derived_and_split():
    """New paired seeds, chosen by a rule rather than after seeing outcomes."""
    doc = json.loads(PROTOCOL.read_text())["behavioural_selection"]["seeds"]
    c0 = json.loads((REPO / "logs/stages/stage-1/phase_c1/plans/"
                            "phase_c0_preregistration.json").read_text())
    c1 = json.loads((REPO / "logs/stages/stage-1/phase_c1/plans/"
                            "execution_preregistration.json").read_text())

    #: Every excluded seed is excluded, and the exclusion set covers BOTH the
    #: Phase-A/B selection seeds and C1's confirmation seeds.
    historical = {int(v) for v in
                  c0["confirmation_seeds"]["historical_seeds_excluded"].values()
                  if isinstance(v, int)}
    c1_seeds = {int(s) for s in c1["seeds"]["values"]}
    assert historical <= set(doc["excluded"])
    assert c1_seeds <= set(doc["excluded"])

    #: Checked on the fields that are USED, not only on the `all` aggregate. A
    #: first version of this test asserted freshness of `all` alone, and a
    #: mutation that poisoned `screening` with a Phase-B seed survived it: the
    #: aggregate stayed clean while the field a launch reads did not.
    forbidden = historical | c1_seeds
    for role in ("screening", "confirmation", "all"):
        assert not (set(doc[role]) & forbidden), (
            f"{role} contains an excluded seed: "
            f"{sorted(set(doc[role]) & forbidden)}")
        assert not (set(doc[role]) & set(doc["excluded"])), role
    #: And the two rungs must ACCOUNT for `all`, so nothing can hide in one.
    assert set(doc["screening"]) | set(doc["confirmation"]) == set(doc["all"])
    assert len(doc["screening"]) + len(doc["confirmation"]) == doc["count"]

    #: Screening and confirmation are disjoint, and sized as the schedule says.
    assert not (set(doc["screening"]) & set(doc["confirmation"]))
    assert len(doc["screening"]) == 1 and len(doc["confirmation"]) == 3
    assert doc["screening_and_confirmation_are_disjoint"] is True

    #: Recomputed from the stated rule, so the values are derived not chosen.
    import hashlib
    base = doc["base_digest"]
    drawn, i = [], 0
    while len(drawn) < doc["count"]:
        value = int.from_bytes(hashlib.sha256(
            f"{base}:phase-c2:recovery-seed:{i}".encode()).digest()[:4],
            "big") % (2 ** 31)
        if value not in doc["excluded"] and value not in drawn:
            drawn.append(value)
        i += 1
    assert drawn == doc["all"], "the seeds are not what the stated rule produces"
    assert doc["bootstrap_seed"] == int.from_bytes(hashlib.sha256(
        f"{base}:phase-c2:bootstrap".encode()).digest()[:4], "big") % (2 ** 31)
    assert base == c1["seeds"]["base_digest"], (
        "C2 must draw from the same frozen base C1 drew from")


def test_the_selection_multiplicity_is_handled_prospectively():
    """Five candidates and one confirmed hypothesis, on disjoint seeds."""
    doc = json.loads(PROTOCOL.read_text())["behavioural_selection"]
    schedule, multiplicity = doc["schedule"], doc["multiplicity"]

    assert schedule["advanced_candidates"] == 1, (
        "more than one advanced candidate requires a multiplicity correction; "
        "the protocol must say which")
    assert schedule["conditional_rung"] is None
    #: Screening ranks and cannot promote.
    assert "no promotion" in schedule["screening"]["decides"].lower()
    assert "only evidence that may name one" in \
        schedule["confirmation"]["decides"].lower()
    #: The claim boundary states the selection explicitly.
    for phrase in ("SELECTED on disjoint screening data", "NOT a simultaneous"):
        assert phrase in multiplicity["claim_boundary"], phrase
    #: The tie-break is outcome-independent of the behavioural data.
    assert "Pareto" in multiplicity["screening_tie_break"]


def test_the_probe_schedule_is_exact_and_b_is_the_only_anchor():
    """B is the comparator. The original control is not a C2 arm."""
    doc = json.loads(PROTOCOL.read_text())
    schedule = doc["behavioural_selection"]["schedule"]
    anchors = doc["behavioural_selection"]["anchors"]

    assert schedule["top_k"] == doc["top_k_selection"]["k"]
    #: B is unconditional in BOTH rungs and is the ONLY anchor in either.
    assert schedule["screening"]["anchors"] == ["frozen_c1_treatment_b"]
    assert schedule["confirmation"]["anchors"] == ["frozen_c1_treatment_b"]
    assert "UNCONDITIONAL" in anchors["frozen_c1_treatment_b"]
    #: And the absence of the old control is DELIBERATE, with a reason, not an
    #: omission — a later turn must not quietly reinstate it.
    assert "_no_canonical_control" in anchors
    assert "canonical_control" not in schedule["confirmation"]["anchors"]
    assert "canonical_control" not in schedule["screening"]["anchors"]
    #: Confirmation is exactly C vs B.
    assert schedule["confirmation"]["arms"] == 2

    #: The counts are arithmetic, not assertions.
    assert schedule["screening"]["probes"] == (
        schedule["top_k"] + len(schedule["screening"]["anchors"])
    ) * schedule["screening"]["seeds"]
    assert schedule["confirmation"]["probes"] == (
        schedule["advanced_candidates"]
        + len(schedule["confirmation"]["anchors"])
    ) * schedule["confirmation"]["seeds"]
    assert schedule["total_probes"] == (schedule["screening"]["probes"]
                                       + schedule["confirmation"]["probes"])


def test_the_catastrophic_veto_operand_is_bound_to_an_arm_that_is_probed():
    """C0 requires the control operand to be named, not silently re-pointed.

    And whatever it names must actually be a probed arm: a veto bound to an arm
    the schedule does not run can never fire.
    """
    doc = json.loads(PROTOCOL.read_text())["behavioural_selection"]
    veto = doc["frozen_science"]["behavioural_guardrails"][
        "catastrophic_capability_veto"]
    assert veto["control_operand"], "the control operand is unbound"
    assert "candidate_operand" in veto
    assert "open_binding_for_c1" not in veto, (
        "C0's open binding must be RESOLVED here, not carried forward unresolved")

    #: Incumbent-relative, as C1's was, with B in that position.
    assert "INCUMBENT" in veto["control_operand"].upper()
    schedule = doc["schedule"]
    assert "frozen_c1_treatment_b" in schedule["confirmation"]["anchors"], (
        "the veto binds the incumbent, so the incumbent must be a confirmation "
        "arm or the veto can never fire")
    #: If it ever names the canonical control again, that arm must be probed.
    if "CANONICAL CONTROL" in veto["control_operand"].upper():
        assert "canonical_control" in schedule["confirmation"]["anchors"]


def test_screening_and_confirmation_score_on_disjoint_prompts():
    """Seed-disjointness alone is insufficient: C0's unit is the prompt.

    C0 measured same-prompt cross-seed dependence (ICC 0.25 +/- 0.095), so
    selecting and confirming on the same prompts would let the selection leak
    into the confirmation through that dependence.
    """
    doc = json.loads(PROTOCOL.read_text())["behavioural_selection"]
    batteries, schedule = doc["batteries"], doc["schedule"]

    assert batteries["screening"]["asset_id"] != batteries["confirmation"]["asset_id"]
    assert (batteries["screening"]["content_sha256"]
            != batteries["confirmation"]["content_sha256"])
    assert schedule["screening"]["battery"] != schedule["confirmation"]["battery"]

    #: Disjointness is a MEASURED zero, not a claim.
    measured = batteries["prompt_disjoint"]["measured"]
    assert measured["shared_ids"] == 0
    assert measured["shared_prompt_hashes"] == 0
    assert set(batteries["prompt_disjoint"]["by"]) == {
        "stable source id", "normalized prompt content hash"}
    #: Same mixture, or a screening delta says nothing about a confirmation one.
    assert batteries["prompt_disjoint"]["mixture_identical"] is True

    #: And the screening asset may not promote anything.
    forbidden = " ".join(batteries["screening"]["may_not"]).lower()
    assert "verdict" in forbidden and "promote" in forbidden


def test_the_screening_battery_record_matches_the_built_asset():
    """The frozen identity must describe the bytes on disk."""
    record = json.loads(
        (REPO / "logs/stages/stage-1/phase_c2/plans/"
                "c2_screening_battery.json").read_text())
    manifest = json.loads(
        (REPO / record["path"].split()[0] / "manifest.json").read_text())
    assert record["content_sha256"] == manifest["content_sha256"]
    assert record["n_prompts"] == manifest["n_prompts"]
    assert record["n_scorable_prompts"] == manifest["n_scorable_prompts"]
    assert record["mixture"] == manifest["mixture"]
    #: Every per-set hash, against the file it names.
    import hashlib
    base = REPO / record["path"].split()[0]
    for name, digest in record["set_sha256"].items():
        assert hashlib.sha256(
            (base / f"{name}.jsonl").read_bytes()).hexdigest() == digest, name
    #: The record must carry its own hash, and it must verify.
    assert record["record_sha256"] == sha256_json(
        {k: v for k, v in record.items() if k != "record_sha256"})


def test_the_screening_sample_was_drawn_under_the_domain_it_claims():
    """The bug this exists for: a manifest claiming a rule it was not drawn under.

    `rank_take` accepted `domain` and dropped it, so the first build of this
    asset was sampled under C1's ordering while its manifest declared
    `phase-c2-screening-battery`. It was still disjoint, still deterministic and
    still exclusion-correct — the only wrong thing was its provenance, which no
    disjointness check can see.

    So this re-derives one stratum from the real source rows and the real
    exclusion sets under the DECLARED domain, and requires the frozen sample to
    be exactly that. It is the claim, checked rather than described.
    """
    import importlib

    sys.path.insert(0, str(REPO / "scripts/data"))
    from battery_render import RENDERERS, rank_take, read_rows
    build = importlib.import_module("build_c2_screening_battery")
    c1_builder = importlib.import_module("build_c1_confirmation_battery")

    record = json.loads(
        (REPO / "logs/stages/stage-1/phase_c2/plans/"
                "c2_screening_battery.json").read_text())
    declared = record["sampling_rule"]["rank_domain"]
    assert declared == build.RANK_DOMAIN

    class _Args:
        battery = "artifacts/eval/battery_v2"
        recovery_search = "artifacts/stage3/recovery_search_v2"
        sessions = "artifacts/stage3/corpus_v2/sessions.jsonl"
        state_eval = "artifacts/stage1/state_eval_v1"
        calibration = "artifacts/stage1/e8_calibration_v1"
        c1_confirmation = build.C1_CONFIRMATION

    for path in (_Args.battery, _Args.sessions, _Args.c1_confirmation):
        if not (REPO / path).exists():
            pytest.skip(f"{path} is not present in this environment")

    ids, hashes, _ = c1_builder.excluded_identities(_Args)
    build.exclude_c1_confirmation(_Args.c1_confirmation, ids, hashes)

    #: One stratum is enough to falsify the provenance claim, and `gsm8k` is the
    #: cheapest to read.
    stratum = "gsm8k"
    want = c1_builder.SETS[stratum][1]
    repo_id, revision, rel = c1_builder.SOURCES[stratum]
    rows = [dict(r, _index=i)
            for i, r in enumerate(read_rows(repo_id, revision, rel))]

    def sample(domain):
        return {str(i["id"]) for i in rank_take(
            rows, want, stratum=stratum, base_digest=c1_builder.C0_DIGEST,
            exclude_ids=ids, exclude_hashes=hashes,
            make=RENDERERS[stratum], domain=domain)}

    frozen = set()
    asset = REPO / record["path"].split()[0] / f"{stratum}.jsonl"
    for line in asset.read_text().splitlines():
        if line.strip():
            frozen.add(str(json.loads(line)["id"]))

    assert sample(declared) == frozen, (
        "the frozen screening sample is not what the declared rank domain "
        "produces; its manifest describes a sampling rule it was not drawn "
        "under")
    #: And it is NOT the default domain's sample, which is what the bug produced.
    from battery_render import DEFAULT_RANK_DOMAIN
    assert sample(DEFAULT_RANK_DOMAIN) != frozen, (
        "the frozen sample equals the DEFAULT-domain sample, which is exactly "
        "the symptom of rank_take dropping its domain argument")


def test_the_screening_ranking_rule_is_frozen_and_uses_no_usable_credit():
    doc = json.loads(PROTOCOL.read_text())["behavioural_selection"]
    screening = doc["schedule"]["screening"]
    rule = screening["ranking_rule"].lower()
    assert "correct_overall" in rule
    assert "candidate - b" in rule or "candidate-b" in rule
    assert "usable_rollout is not positive ranking credit" in rule
    #: B is an anchor, never something that can advance.
    assert "anchor, never an advancing candidate" in rule
    #: The tie-break is fixed before any behavioural datum exists.
    assert "state id" in screening["tie_break"].lower()
    assert "before any" in screening["tie_break"].lower()


def test_the_interpretation_boundary_separates_search_from_recovery():
    """A cheap-metric front is not a capability claim."""
    doc = json.loads(PROTOCOL.read_text())["interpretation_boundary"]
    search = doc["the_search_stages_train_nothing"]
    assert "NO 0.86M recovery training" in search["what_they_do_not_do"]
    assert "SELECTION evidence only" in search["status_of_their_output"]
    assert "may never promote" in search["status_of_their_output"]

    #: C1's level is derived from its committed probe results, not restated.
    post = doc["c1_incumbent_is_a_post_recovery_measurement"]
    probes = json.loads((REPO / post["source"]).read_text())["probes"]
    import collections
    pooled = collections.defaultdict(lambda: [0, 0])
    for probe in probes:
        pooled[probe["arm"]][0] += probe["counts"]["correct"]
        pooled[probe["arm"]][1] += probe["counts"]["n_scorable"]
    assert post["b_after_c1"]["correct"] == pooled["treatment"][0]
    assert post["b_after_c1"]["scorable"] == pooled["treatment"][1]
    assert post["incumbent_before_c1"]["correct"] == pooled["incumbent"][0]
    assert post["b_after_c1"]["correct_overall"] == pytest.approx(
        pooled["treatment"][0] / pooled["treatment"][1], abs=1e-8)
    assert "NOT raw initialization accuracy" in post["_what_these_numbers_are"]

    #: And promotion depends only on the fresh recovery comparison.
    assert "ONLY the fresh recovery comparison" in doc[
        "what_c2_promotion_depends_on"]
    assert "does C improve on B" in doc["the_target_is_the_current_incumbent"]


# --- 7. the documents are plans, and say so --------------------------------


@pytest.mark.parametrize("path,field", [
    (PROTOCOL, "protocol_sha256"), (PRICING, "pricing_sha256")])
def test_each_document_self_hashes_and_authorizes_nothing(path, field):
    doc = json.loads(path.read_text())
    assert doc["authorizes"] == "nothing"
    recomputed = sha256_json({k: v for k, v in doc.items() if k != field})
    assert doc[field] == recomputed, f"{path.name}: {field} does not verify"


def test_the_protocol_withdraws_search_2_and_freezes_search_1():
    doc = json.loads(PROTOCOL.read_text())
    assert "WITHDRAWN" in doc["supersedes"]["decision"]
    assert "FROZEN" in doc["supersedes"]["search_1_status"]
    for forbidden in ("rerunning Search-1 or the Search-1 beam",
                      "remeasuring B or any frozen C candidate"):
        assert forbidden in doc["explicitly_not_authorized"], forbidden


def test_the_protocol_does_not_pin_any_calibration():
    doc = json.loads(PROTOCOL.read_text())
    assert doc["space"]["impl_profiles"] is None
    assert doc["beam_and_ranking"]["standing_width"] > 0


def test_the_documents_are_deterministic_and_regenerating_verifies_them():
    """Regenerating must produce IDENTICAL bytes, so a reviewer can check them.

    The first draft of these documents carried a `generated_utc`, which moved
    the self-hash on every run and made regeneration a new document rather than
    a verification. This project has already had a pricing hash become a bound
    identity that a launch verifies, so an unreproducible plan hash is a trap
    rather than an inconvenience.
    """
    import subprocess

    before = {path: path.read_bytes() for path in (PROTOCOL, PRICING)}
    result = subprocess.run(
        [sys.executable, "scripts/autoinit/write_c2_full_search_plan.py",
         "--write"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": "src:scripts:scripts/autoinit", "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr[-2000:]
    for path, original in before.items():
        assert path.read_bytes() == original, (
            f"{path.name} changed when regenerated from the same tree; the "
            "generator is not deterministic and its self-hash cannot be "
            "reproduced")
    #: And no clock field crept back in.
    for path in (PROTOCOL, PRICING):
        assert "generated_utc" not in json.loads(path.read_text())

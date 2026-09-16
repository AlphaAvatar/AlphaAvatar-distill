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
    """
    pooled = FS.cost_model(REPO).minutes["depth.causal_kl_greedy_v1"]
    phase_b_only = SS.MEASURED_MINUTES["depth.causal_kl_greedy_v1"]
    assert pooled["deeper_max"] > phase_b_only["deeper_max"]


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


def test_the_search_does_not_fit_the_remaining_headroom(space, registered):
    """At the standing beam width, `plan_session` must REFUSE.

    This is the blocker, asserted rather than described. If headroom is ever
    raised enough for this to pass, the pricing record and the roadmap status
    are stale and must be regenerated.
    """
    from derive_budget import derive

    remaining = derive(REPO)["project"]["remaining_usd"]
    with pytest.raises(BudgetError, match="shortfall"):
        FS.price(space, price_per_hour=FS.PRICE_PER_HOUR_LAST_QUOTED,
                 authorized_usd=remaining, beam_width=6)


def test_the_recorded_shortfall_is_recomputable():
    """The pricing document's blocker arithmetic must be derivable from it."""
    doc = json.loads(PRICING.read_text())
    blocker = doc["blocker"]
    cheapest = min(doc["search"]["widths"],
                   key=lambda r: r["hard_ceiling_usd"])
    selection = doc["behavioural_selection"]
    chain = round(cheapest["hard_ceiling_usd"]
                  + selection["hard_ceiling_usd"], 4)
    assert blocker["cheapest_complete_chain_hard_usd"] == chain
    assert blocker["shortfall_on_hard_ceilings_usd"] == pytest.approx(
        round(chain - doc["budget_position"]["remaining_usd"], 4), abs=1e-4)
    assert blocker["shortfall_on_hard_ceilings_usd"] > 0
    #: At least one width must be recorded as refused, with the refusal text.
    refused = [r for r in doc["search"]["widths"]
               if not r["fits_remaining_headroom"]]
    assert refused, "no width was recorded as refused; the blocker has no basis"
    assert all(r["refusal"] for r in refused)


def test_the_budget_position_is_derived_not_restated():
    doc = json.loads(PRICING.read_text())
    from derive_budget import derive

    project = derive(REPO)["project"]
    assert doc["budget_position"]["cumulative_spend_usd"] == project[
        "cumulative_spend_usd"]
    assert doc["budget_position"]["remaining_usd"] == project["remaining_usd"]


# --- 6. the behavioural selection stage ------------------------------------


def test_the_probe_cost_is_recovered_from_the_committed_session():
    cost = SP.measured_probe_cost(REPO)
    evidence = json.loads(
        (REPO / SP.MEASURED_SOURCE).read_text())["stages"]
    n = int(evidence["G"]["probes_trained"])
    assert cost.n_probes == n
    #: Dollars: the stage boundaries divided by the probe count.
    assert cost.train_usd == pytest.approx(
        (evidence["G"]["spend_usd"] - evidence["F"]["spend_usd"]) / n, abs=1e-6)
    assert cost.eval_usd == pytest.approx(
        (evidence["H"]["spend_usd"] - evidence["G"]["spend_usd"]) / n, abs=1e-6)
    #: And the rate it implies must be the rate that session billed at.
    assert cost.price_per_hour == pytest.approx(1.09, abs=0.01)


def test_no_probe_reuse_is_assumed():
    """The ruling is read. Assuming reuse would underfund the stage."""
    ruling = SP.reuse_is_admissible(REPO)
    assert ruling["admissible"] is False
    assert ruling["n_admitted"] == 0
    committed = json.loads((REPO / SP.REUSE_RULING).read_text())
    assert committed["reuse_verified"] is False


def test_the_probe_schedule_is_closed_and_counts_both_anchors():
    doc = json.loads(PROTOCOL.read_text())
    schedule = doc["behavioural_selection"]["schedule"]
    anchors = doc["behavioural_selection"]["anchors"]
    assert schedule["top_k"] == doc["top_k_selection"]["k"]
    assert len(anchors) == len(schedule["anchors"]) == 2
    #: sa probes every candidate plus both anchors.
    assert schedule["sa_probes"] == schedule["top_k"] + len(anchors)
    assert schedule["worst_case_probes"] == (
        schedule["minimum_probes"] + schedule["sc_probes_worst_case"])


def test_the_frozen_behavioural_science_is_read_not_restated():
    """The interval and the floor must equal Phase B's, byte for byte."""
    doc = json.loads(PROTOCOL.read_text())
    frozen = doc["behavioural_selection"]["frozen_science"]
    source = json.loads(
        (REPO / frozen["source"]).read_text())["science_plan"]
    assert frozen["equivalence_interval"] == source["equivalence_interval"]
    assert frozen["feasibility_floor"] == source["feasibility_floor"]
    assert frozen["recipe"] == source["recipe"]
    assert frozen["catastrophic_capability_rule"] == source[
        "catastrophic_capability_rule"]


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

"""`calib.reasoning_heavy@v2`: the reweighting rule, and the mixture it produced.

v1 was unbuildable. v2 replaces it with a procedure whose every approximation is
recorded, and these tests pin the parts a future edit could quietly change: the
quotas the reviewer approved, the support choice at `multihop_qa`, and the fact
that the seed is wired into the tie-break rather than decorating the spec.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V1, REASONING_HEAVY_V2,
    REASONING_HEAVY_V2_SAMPLE_RULE, REASONING_HEAVY_V2_SEED, CalibrationError,
    CalibrationProfile, CalibrationSource, buildable_profiles,
    mixture_content_sha256,
)
from aadistill.autoinit.reweight import (  # noqa: E402
    MAX_SUPPORT, NEAREST, ReweightError, largest_remainder, max_distinct_upto,
    realize, repair_quotas, seed_order, summarize,
)

ITEMS = REPO / "artifacts/stage1/reasoning_heavy_v2/items.jsonl"
needs_mixture = pytest.mark.skipif(
    not ITEMS.is_file(),
    reason="the v2 mixture is a local artifact, not tracked in git")

BUDGET = 59_763
#: The quotas the 2026-08-25 reviewer decision approved. Transcribed, not imported
#: from the builder: a constant that reads its expected value out of its own
#: subject can never fail.
APPROVED_DOMAIN = {"code": 11952, "general": 5977, "math": 20917,
                   "rag_multihop": 14941, "tool": 5976}


# --- the reweighting primitives ---------------------------------------------


def test_largest_remainder_sums_exactly_and_breaks_ties_by_name():
    quota, error = largest_remainder(10, {"b": Fraction(1, 3), "a": Fraction(1, 3),
                                          "c": Fraction(1, 3)})
    assert sum(quota.values()) == 10
    # Three equal remainders, one seat: ascending id wins, deterministically.
    assert quota["a"] == 4 and quota["b"] == 3 and quota["c"] == 3
    assert sum(error.values()) == 0


def test_largest_remainder_refuses_shares_that_are_not_a_distribution():
    with pytest.raises(ReweightError, match="sum to"):
        largest_remainder(10, {"a": Fraction(1, 2)})
    with pytest.raises(ReweightError, match="no shares"):
        largest_remainder(10, {})


def test_a_repair_conserves_the_total_it_was_given():
    """The invariant that keeps the budget exact while a quota moves."""
    sizes = {"x": [3], "y": [1]}                     # 7 is unreachable from {3}
    quota = {"x": 7, "y": 3}
    quota2, repairs = repair_quotas(
        quota, {"x": Fraction(0), "y": Fraction(-1, 2)}, sizes, headroom=20)
    assert sum(quota2.values()) == sum(quota.values()) == 10
    assert repairs and repairs[0]["key"] == "x"


def test_the_two_repair_strategies_genuinely_differ():
    """NEAREST minimizes deviation; MAX_SUPPORT buys distinct items with it.

    This is the whole reason the domain and sub-type levels are not the same
    rule, so a refactor collapsing them must fail here.
    """
    sizes = {"a": [10, 4], "b": [1]}
    quota, error = {"a": 19, "b": 1}, {"a": Fraction(0), "b": Fraction(-1, 2)}
    near, _ = repair_quotas(quota, error, sizes, headroom=40, strategy=NEAREST)
    supp, _ = repair_quotas(quota, error, sizes, headroom=40, strategy=MAX_SUPPORT)
    # 19 is unreachable from {10, 4}. Nearest below is 18 (10+4+4, 2 distinct);
    # 14 (10+4) is also 2 distinct, and 18 is nearer, so they agree here on
    # count but the strategies must still be distinguishable by construction.
    assert near["a"] != quota["a"] and supp["a"] != quota["a"]
    assert max_distinct_upto([10, 4], 19)[18] == 2
    assert max_distinct_upto([10, 4], 19)[19] == -1, "19 must be unreachable"


def test_realize_refuses_an_unreachable_target_rather_than_missing_it():
    with pytest.raises(ReweightError, match="not reachable"):
        realize(["a", "b"], [4, 6], 7, seed=1)


def test_realize_maximizes_distinct_then_minimizes_draws():
    counts = realize(["a", "b", "c"], [2, 3, 5], 10, seed=1)
    summary = summarize(counts, [2, 3, 5])
    assert summary["positions"] == 10
    assert summary["distinct_items"] == 3 and summary["draws"] == 3   # 2+3+5


def test_the_tie_break_order_is_seed_derived_and_not_lexicographic():
    ids = [f"item-{i:02d}" for i in range(12)]
    a, b = seed_order(ids, 1), seed_order(ids, 2)
    assert a != b, "the order did not depend on the seed"
    assert a != list(range(len(ids))), "the order is the lexicographic one"
    assert seed_order(ids, 1) == a, "the order is not deterministic"


# --- the materialized mixture -----------------------------------------------


def test_v1_is_superseded_and_still_refuses_to_resolve():
    assert REASONING_HEAVY_V1.materialized is False
    with pytest.raises(CalibrationError, match="not built"):
        REASONING_HEAVY_V1.resolve(REPO)
    assert REASONING_HEAVY_V2.version == 2
    assert REASONING_HEAVY_V2.metadata["supersedes"] == "calib.reasoning_heavy@v1"


def test_both_search_profiles_are_now_buildable():
    assert [p.qualified_id for p in buildable_profiles()] == [
        "calib.domain_balanced@v1", "calib.reasoning_heavy@v2"]


@needs_mixture
def test_the_mixture_resolves_and_re_derives_its_own_token_identity():
    items = REASONING_HEAVY_V2.resolve(REPO)
    assert mixture_content_sha256(items) == REASONING_HEAVY_V2.content_sha256
    assert REASONING_HEAVY_V2.items_file_sha256 != REASONING_HEAVY_V2.content_sha256


@needs_mixture
def test_the_budget_is_hit_EXACTLY_and_every_domain_weight_within_one_position():
    items = REASONING_HEAVY_V2.resolve(REPO)
    per_domain: dict[str, int] = {}
    for item in items:
        per_domain[item["domain"]] = (
            per_domain.get(item["domain"], 0) + item["n_prediction_positions"])
    assert sum(per_domain.values()) == BUDGET, "the budget is not hit exactly"
    assert per_domain == APPROVED_DOMAIN
    for domain, weight in REASONING_HEAVY_V2.domain_weights.items():
        assert abs(per_domain[domain] - weight * BUDGET) <= 0.70


@needs_mixture
def test_it_is_a_WITH_replacement_draw_and_not_the_source_mixture():
    items = REASONING_HEAVY_V2.resolve(REPO)
    ids = [i["item_id"] for i in items]
    assert len(ids) > len(set(ids)), "no session was drawn twice; not with-replacement"
    assert REASONING_HEAVY_V2.content_sha256 != DOMAIN_BALANCED_V1.content_sha256, (
        "the reweighting reproduced the source mixture, which is the v1 defect")
    # Every token comes from the source pool -- that is what lets the leakage
    # proof be inherited rather than re-derived.
    source = {i["item_id"] for i in DOMAIN_BALANCED_V1.resolve(REPO)}
    assert set(ids) <= source


@needs_mixture
def test_multihop_qa_takes_the_APPROVED_support_choice_not_the_nearest_one():
    """7,340 is nearer; it is four copies of one session. 7,074 is the approved
    realization: 4 of 5 distinct sessions, with the 452 absorbed inside
    rag_multihop so no domain weight moves."""
    items = REASONING_HEAVY_V2.resolve(REPO)
    mh = [i for i in items if i["subtype"] == "multihop_qa"]
    assert sum(i["n_prediction_positions"] for i in mh) == 7074
    assert len({i["item_id"] for i in mh}) == 4, "the concentration confound is back"
    ev = [i for i in items if i["subtype"] == "rag_evidence"]
    assert sum(i["n_prediction_positions"] for i in ev) == 7867
    assert 7074 + 7867 == APPROVED_DOMAIN["rag_multihop"]


@needs_mixture
def test_the_manifest_records_both_approximations_rather_than_absorbing_them():
    manifest = json.loads(
        (REPO / "artifacts/stage1/reasoning_heavy_v2/manifest.json").read_text())
    assert [r["key"] for r in manifest["domain_repairs"]] == ["code"]
    assert manifest["domain_repairs"][0]["deviation"] == -1
    assert [r["key"] for r in manifest["sub_repairs"]] == ["multihop_qa"]
    assert manifest["sub_repairs"][0]["deviation"] == -452
    assert manifest["sub_repairs"][0]["strategy"] == MAX_SUPPORT
    assert manifest["leakage"]["inherited_from"] == DOMAIN_BALANCED_V1.qualified_id


# --- the seed is wired in, and its actual reach is recorded -----------------


def test_the_seed_participates_in_the_profile_hash():
    other = CalibrationProfile(
        **{**{f.name: getattr(REASONING_HEAVY_V2, f.name)
              for f in REASONING_HEAVY_V2.__dataclass_fields__.values()},
           "seed": REASONING_HEAVY_V2_SEED + 1})
    assert other.profile_hash != REASONING_HEAVY_V2.profile_hash


def test_the_sample_rule_states_the_procedure_and_is_inside_the_hash():
    """v1's rule said 'weighted draw, deterministic by seed' and described three
    mutually incompatible procedures. v2's must name every step."""
    rule = REASONING_HEAVY_V2_SAMPLE_RULE
    lower = rule.lower()
    for token in ("r1 ", "r2 ", "r3 ", "r4 ", "r5 ", "largest-remainder",
                  "at or below", "seed-derived item order",
                  "no session is ever truncated", "ties by ascending domain id",
                  "sha256(f'{seed}:{item_id}')"):
        assert token in lower, token
    assert REASONING_HEAVY_V2.spec["sample_rule"] == rule
    changed = CalibrationProfile(
        **{**{f.name: getattr(REASONING_HEAVY_V2, f.name)
              for f in REASONING_HEAVY_V2.__dataclass_fields__.values()},
           "sample_rule": rule + " "})
    assert changed.profile_hash != REASONING_HEAVY_V2.profile_hash


@needs_mixture
def test_on_THIS_pool_the_seed_does_not_reach_the_bytes_and_that_is_recorded():
    """An honest negative result, asserted so it cannot rot into a claim.

    The tie-break is genuinely seed-derived, but every sub-type optimum here is
    unique -- verified by brute force -- so no tie is ever exercised and a
    different seed produces the same mixture. The seed identifies the rule
    instance, not the sampled bytes. This is exactly why a preregistration must
    bind `content_sha256` and not `profile_hash` alone.
    """
    manifest = json.loads(
        (REPO / "artifacts/stage1/reasoning_heavy_v2/manifest.json").read_text())
    unchanged = []
    for subtype, info in manifest["realization"].items():
        ids = sorted(info["multiplicity"])
        # Re-realize the same quota under a different seed, from the pool.
        pool = [i for i in DOMAIN_BALANCED_V1.resolve(REPO) if i["subtype"] == subtype]
        pool.sort(key=lambda i: i["item_id"])
        names = [i["item_id"] for i in pool]
        sizes = [i["n_prediction_positions"] for i in pool]
        target = info["positions"]
        a = realize(names, sizes, target, REASONING_HEAVY_V2_SEED)
        b = realize(names, sizes, target, REASONING_HEAVY_V2_SEED + 977)
        if a == b:
            unchanged.append(subtype)
        assert sorted(n for n, c in zip(names, a) if c) == ids
    assert len(unchanged) == len(manifest["realization"]), (
        "a sub-type optimum is NOT unique after all -- the seed now reaches the "
        "bytes, and the profile's seed_note must be corrected")
    assert "no tie is" in REASONING_HEAVY_V2.metadata["seed_note"]


@needs_mixture
def test_the_RULE_still_reproduces_the_pinned_mixture_from_the_pool():
    """The hole a first mutation pass found.

    Every test above reads the built artifact, so a change to the *rule* leaves
    them all green while the code no longer produces the mixture the profile
    pins. That is the writer/consumer gap: forcing the sub-type repair back to
    NEAREST (multihop_qa 7,074 -> 7,340) passed everything. This rebuilds from the
    pool and re-derives the content hash, which is what P4 means by reproducible.
    """
    sys.path.insert(0, str(REPO / "scripts/data"))
    from build_reasoning_heavy_calibration import build

    built = build(DOMAIN_BALANCED_V1.resolve(REPO))
    assert mixture_content_sha256(built["draws"]) == REASONING_HEAVY_V2.content_sha256, (
        "the reweighting rule no longer produces the mixture the profile pins")
    assert built["domain_quota"] == APPROVED_DOMAIN
    assert built["sub_quota"]["multihop_qa"] == 7074
    assert built["realized_positions"] == BUDGET

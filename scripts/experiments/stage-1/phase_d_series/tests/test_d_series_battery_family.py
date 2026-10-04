"""The D-series battery family: frozen prospectively, disjoint by construction.

The family exists because `c1_confirmation_v1` must stay held out and the pool
under the frozen C1 mixture is exhausted. Its whole value is that the allocation
was fixed BEFORE any D1 outcome existed, so these tests are mostly about what
cannot move and what cannot be decided late.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.phase_d_series import battery_family as family  # noqa: E402


class TestTheSixRoles:

    def test_there_are_six_in_a_fixed_order(self):
        assert family.N_ROLES == 6
        assert [r for r, _d, _e, _p in family.ROLES] == [
            "d1_screening", "d1_confirmation",
            "d2_screening", "d2_confirmation",
            "d3_screening", "d3_confirmation"]

    def test_every_experiment_gets_a_screening_and_a_confirmation_role(self):
        by_experiment: dict[str, set[str]] = {}
        for role, _domain, experiment, _purpose in family.ROLES:
            by_experiment.setdefault(experiment, set()).add(
                role.split("_", 1)[1])
        assert by_experiment == {"D1": {"screening", "confirmation"},
                                 "D2": {"screening", "confirmation"},
                                 "D3": {"screening", "confirmation"}}

    def test_the_rank_domains_are_distinct(self):
        """A shared domain would give two roles the same ordering of the same
        pool, so the second would be the first's prefix rather than an
        independent sample."""
        domains = [d for _r, d, _e, _p in family.ROLES]
        assert len(set(domains)) == len(domains)

    def test_the_rank_domains_are_pinned(self):
        """`rank_key` HASHES these strings. A typo produces a different sample
        of the same pool rather than an error, so the exact strings are the
        thing under test."""
        assert [d for _r, d, _e, _p in family.ROLES] == [
            "d-series-v1-d1-screening", "d-series-v1-d1-confirmation",
            "d-series-v1-d2-screening", "d-series-v1-d2-confirmation",
            "d-series-v1-d3-screening", "d-series-v1-d3-confirmation"]

    def test_none_of_them_collides_with_a_historical_rank_domain(self):
        """C1 ordered its battery under `phase-c1-battery` and C2 under
        `phase-c2-screening-battery`. A D-series role reusing either would draw
        the same ordering a consumed battery was drawn from."""
        import battery_render
        import build_c2_screening_battery as c2

        historical = {battery_render.DEFAULT_RANK_DOMAIN, c2.RANK_DOMAIN}
        assert not historical & {d for _r, d, _e, _p in family.ROLES}


class TestDisjointnessIsNotAssumed:

    def test_the_exclusion_chain_grows_with_the_build_order(self):
        """Distinct rank domains give INDEPENDENT samples, not disjoint ones:
        two independent draws from one pool overlap. Disjointness comes from
        each role excluding every role built before it."""
        chain = family.exclusion_chain()["chain"]
        sizes = [len(link["excludes"]) for link in chain]
        assert sizes == sorted(sizes)
        assert sizes[-1] - sizes[0] == family.N_ROLES - 1
        #: The last role excludes all five earlier ones by name.
        last = chain[-1]["excludes"]
        for role, _d, _e, _p in family.ROLES[:-1]:
            assert role in last
        assert chain[0]["excludes"].count("d1_screening") == 0

    def test_both_held_out_behavioural_batteries_are_excluded_by_every_role(self):
        for link in family.exclusion_chain()["chain"]:
            for battery in family.HELD_OUT_BATTERIES:
                assert battery in link["excludes"]

    def test_c1_confirmation_is_one_of_them(self):
        """The reason the family exists: B was promoted on it and C2's, C3's and
        A3's results were measured on it, so it stays held out."""
        assert any("c1_confirmation_v1" in b
                   for b in family.HELD_OUT_BATTERIES)

    def test_all_three_identity_kinds_are_required(self):
        """Two was insufficient, and this test said two.

        v1 bound a stable id and a rendered-prompt hash. The 2026-10-03 source
        round measured what they miss: MBPP full/train 602 is the same problem as
        consumed full/test 217 and differs in both, and the historical chain's
        training-corpus hash catches 0 of the 1,708 GSM8K train rows in the
        corpus. The third coordinate is canonical problem content.
        """
        chain = family.exclusion_chain()
        assert len(chain["identity_kinds"]) == 3
        kinds = " ".join(chain["identity_kinds"]).lower()
        assert "native/source identity" in kinds
        assert "rendered-prompt exact identity" in kinds
        assert "problem-content identity" in kinds
        assert "misses" in chain["all_three_required"]


class TestItIsFrozenProspectively:

    def test_the_rule_id_is_computable_without_any_pool(self):
        """Which is what "frozen before any D1 outcome" means operationally:
        the hash exists today, with no source pinned and no item drawn."""
        assert len(family.allocation_rule_id()) == 32
        assert family.allocation_rule_id() == family.allocation_rule_id()

    def test_the_rule_id_moves_with_the_role_order(self, monkeypatch):
        before = family.allocation_rule_id()
        monkeypatch.setattr(family, "ROLES", tuple(reversed(family.ROLES)))
        assert family.allocation_rule_id() != before

    def test_the_rule_id_moves_with_a_rank_domain(self, monkeypatch):
        before = family.allocation_rule_id()
        tampered = list(family.ROLES)
        role, _domain, experiment, purpose = tampered[0]
        tampered[0] = (role, "d-series-v1-TAMPERED", experiment, purpose)
        monkeypatch.setattr(family, "ROLES", tuple(tampered))
        assert family.allocation_rule_id() != before

    def test_the_rule_id_moves_with_the_mixture(self, monkeypatch):
        before = family.allocation_rule_id()
        monkeypatch.setattr(family, "strata",
                            lambda: {"gsm8k": ("reasoning_math", 1, True)})
        assert family.allocation_rule_id() != before

    def test_the_content_id_follows_the_realization_rather_than_being_typed(self):
        """It was None while nothing was built, and that was correct then.

        The family is built now, so a null here would be the opposite error -- a
        design record claiming 22 MB of evaluation data does not exist while the
        manifest and the state snapshot say it does. `_realization()` derives it
        from whether the manifest exists, so building or removing the family
        moves this without an edit.
        """
        report = family.report()
        manifest = REPO / "logs/shared/analyses/autoinit_d_series_family_manifest.json"
        if manifest.is_file():
            import json as _json

            realized = _json.loads(manifest.read_text())
            assert report["status"] == "BUILT / VERIFIED"
            assert report["family_content_id"] == realized["family_content_id"]
            assert report["realization"]["owner"].endswith(
                "autoinit_d_series_family_manifest.json")
            assert report["capacity_source_blocker"] == "CLOSED"
        else:
            assert report["family_content_id"] is None
            assert "fabricated" in report["_family_content_id_is_null"]

    def test_the_rule_excludes_only_what_it_cannot_know_yet(self):
        """The line moved, for a stated reason.

        v1 kept every source name out of the rule because the sources were an
        open decision. They are now DECIDED, and which population a stratum
        draws from determines its candidate pool — so it decides which rows can
        win the ranking, and it belongs in the prospective rule.

        What still may not appear: the per-file DIGESTS and the realized item
        lists. Those describe what was drawn rather than how, and binding them
        would make the rule unfreezable until materialization.
        """
        rule = family.allocation_rule()
        rendered = json.dumps(rule)
        #: a source POLICY is in; a per-file digest or a realized item is not
        assert "source_policy" in rule
        for forbidden in ("parquet", "sha256_of_file", "realized_items",
                          "item_list"):
            assert forbidden not in rendered.lower(), forbidden
        #: and a 64-hex file digest must not have crept in anywhere
        import re as _re
        assert not _re.search(r"\b[0-9a-f]{64}\b", rendered), (
            "a file digest is in the rule; digests belong to family_content_id")
        #: `sha256` DOES appear — in the order formula, which is the mechanism
        #: and not a pin. The distinction is the point: the rule may describe
        #: how it will hash, and may not name what it will hash over.
        assert "sha256" in rule["selection"]["order"].lower()
        #: v1 said "source pins" were out. v2 keeps the per-file DIGESTS out and
        #: the POLICY in, which is the correction.
        assert "DIGESTS" in rule["what_is_NOT_in_this_rule"]
        assert "realized item lists" in rule["what_is_NOT_in_this_rule"]
        #: the REASON, not a phrase. v1 said "unfreezable"; v2 says the excluded
        #: things describe what was drawn rather than how. Pinning the wording
        #: makes a test fail on a rewrite that improved it.
        excluded_why = rule["what_is_NOT_in_this_rule"]
        assert "family_content_id" in excluded_why
        assert "what was drawn rather than how" in excluded_why
        assert "source pins" not in excluded_why, (
            "the sources are decided; calling their pins unknowable is stale")

    def test_there_is_no_outcome_dependence_and_it_says_so(self):
        selection = family.allocation_rule()["selection"]
        assert selection["deterministic"] is True
        assert selection["outcome_dependence"].startswith("NONE")
        #: No seed and no date anywhere in the rule: both are choices available
        #: after a result is seen.
        rendered = json.dumps(family.allocation_rule()).lower()
        assert "seed" not in rendered.replace("no seed", "")


class TestItIsANewDistributionAndSaysSo:

    def test_the_family_id_is_not_c1s(self):
        assert family.FAMILY_ID == "d_series_behavioural_v1"
        identity = family.allocation_rule()["distribution_identity"]
        assert identity["is_not"] == "c1_confirmation"

    def test_the_stratum_balance_is_inherited_not_restated(self):
        """Imported from the C1 builder, so the two cannot drift apart."""
        import build_c1_confirmation_battery as c1

        assert family.strata() == dict(c1.SETS)
        rule = family.allocation_rule()
        assert rule["n_prompts_per_role"] == 950
        assert rule["n_scorable_prompts_per_role"] == 850
        assert "imported" in rule["mixture_source"]

    def test_the_population_change_is_the_stated_reason_for_a_new_identity(self):
        why = family.allocation_rule()["distribution_identity"]["why"].lower()
        assert "different population" in why
        assert "not comparable" in why
        assert "not imported" in why

    def test_the_sesoi_is_carried_as_an_assumption(self):
        transfer = family.allocation_rule()["distribution_identity"][
            "sesoi_transfer"]
        assert "ASSUMPTION" in transfer
        assert "characterized on the C1 population" in transfer

    def test_a_score_on_this_family_may_not_be_claimed_comparable(self):
        claims = family.report()["what_this_may_not_be_used_to_claim"]
        joined = " ".join(claims)
        assert "comparable with a C1 score" in joined
        #: the enduring claim boundary. "None is built" was the right thing to
        #: forbid while none was; with six built, what must not be claimed is a
        #: RESULT -- the batteries are data and nothing has been measured on them.
        assert "nothing has been measured on them" in joined


class TestTheRequirementNobodyHadComputed:

    def test_six_roles_need_six_times_the_mixture(self):
        for stratum, row in family.requirement().items():
            assert row["needed_for_six_roles"] == row["per_battery"] * 6, \
                stratum

    def test_three_strata_are_short_not_one(self):
        """The finding. `d1_evidence_capacity` asks how many whole batteries
        remain and answers zero, binding on `math_verified`. At six roles
        `gsm8k` and `code` are short too, so extending math alone would unblock
        D1 and leave the family short."""
        short = family.report()["short_strata"]
        assert set(short) == {"math_verified", "code", "gsm8k"}
        assert short["math_verified"] == 830
        assert short["code"] == 321
        assert short["gsm8k"] == 11

    def test_the_other_four_strata_fund_six_roles(self):
        need = family.requirement()
        for stratum in ("knowledge", "multihop", "rag", "tool"):
            assert need[stratum]["short_by"] == 0
            assert need[stratum]["roles_fundable_today"] >= 6

    def test_it_reads_the_committed_capacity_record(self):
        """Rather than re-deriving: a second derivation needs the HF snapshots
        and could silently disagree with the record the blocker is stated
        from."""
        record = json.loads(
            (REPO / family.CAPACITY_RECORD).read_text())
        need = family.requirement()
        for stratum, row in record["strata"].items():
            assert need[stratum]["eligible_today"] == row["eligible"]

    def test_the_disagreement_with_that_record_is_explained(self):
        report = family.report()
        why = report["capacity_record"]["_why_the_two_disagree"]
        assert "six roles" in why
        assert report["capacity_record"]["batteries_remaining_for_one"] == 0

    def test_the_blocker_is_no_longer_about_capacity(self):
        """It was "THREE strata cannot fund six roles". The source decision
        closed that, so a blocker still naming it would be describing a state the
        repository has left -- and the per-stratum shortfalls move to their own
        field, because what they are now is the history of a closed problem."""
        report = family.report()
        blocker = report["blocker"]
        assert "CAPACITY IS CLOSED" in blocker
        assert "RENDERER" in blocker and "MATERIALIZATION" in blocker
        assert "No row is drawn" in blocker
        #: the shortfalls are kept, where they belong
        short = report["shortfall_against_the_original_pins"]
        for stratum in ("math_verified", "code", "gsm8k"):
            assert stratum in short
        #: and the record that owns the eligible figures is named rather than
        #: the figures being copied here
        assert "source_evidence.json" in blocker


class TestTheSourceOptionsArePinnedToNothing:

    def test_every_short_stratum_has_at_least_one_option(self):
        short = family.report()["short_strata"]
        for stratum in short:
            assert family.SOURCE_OPTIONS.get(stratum), stratum

    def test_no_option_asserts_a_revision_or_a_digest(self):
        """Asserting provenance without fetching and hashing it is inventing
        it. The options say what must be measured instead."""
        for stratum, options in family.SOURCE_OPTIONS.items():
            for option in options:
                assert "revision" not in option, stratum
                assert "sha256" not in json.dumps(option), stratum
                assert option["must_be_measured"]

    def test_an_unverified_licence_says_so(self):
        for options in family.SOURCE_OPTIONS.values():
            for option in options:
                if not option["same_repo"]:
                    assert "VERIFIED" in option["licence"].upper()

    def test_the_cheapest_options_reuse_an_already_pinned_repository(self):
        """gsm8k is short by 11 items and MBPP has unread splits: both can be
        answered by another FILE of a repository already pinned at a frozen
        revision, which inherits its licence and its renderer."""
        assert any(o["same_repo"] for o in family.SOURCE_OPTIONS["gsm8k"])
        assert any(o["same_repo"] for o in family.SOURCE_OPTIONS["code"])

    def test_the_code_option_names_the_risk_that_it_is_not_enough(self):
        first = family.SOURCE_OPTIONS["code"][0]
        assert "may NOT cover" in first["must_be_measured"]
        assert "usable_rollout_rate" in first["must_be_measured"]


class TestTheCommittedRecord:

    def test_it_regenerates_byte_identically(self):
        path = REPO / family.RECORD
        assert path.is_file(), "run battery_family.py --write"
        assert (json.dumps(family.report(), indent=1, sort_keys=True) + "\n") \
            == path.read_text()

    def test_it_materializes_and_authorizes_nothing(self):
        doc = json.loads((REPO / family.RECORD).read_text())
        assert doc["_authorizes"] == "nothing"
        assert doc["_materializes"] == "nothing", (
            "this PRODUCER materializes nothing -- build_batteries.py does")
        assert doc["status"] in ("BUILT / VERIFIED", "DESIGNED / NOT MATERIALIZED")

    def test_it_lives_in_the_shared_area_not_under_one_experiment(self):
        """Three experiments own two roles each. Filing it under `phase_d1`
        would make D1's directory the authority on D2's evidence."""
        assert family.RECORD.startswith("logs/shared/analyses/")
        assert "phase_d" not in family.RECORD

    def test_it_is_tracked_rather_than_ignored(self):
        import subprocess

        done = subprocess.run(["git", "check-ignore", family.RECORD],
                              cwd=REPO, capture_output=True, text=True)
        assert done.returncode != 0, "the record would not be committed"


class TestZeroRolesAreFundableToday:

    def test_the_family_cannot_be_built_and_the_report_leads_with_that(self):
        report = family.report()
        assert report["roles_fundable_today"] == 0
        assert report["binding_stratum"] == "math_verified"

    @pytest.mark.parametrize("key", ["allocation_rule", "allocation_rule_id",
                                     "requirement", "short_strata",
                                     "source_options", "blocker"])
    def test_the_report_is_complete_without_a_single_battery(self, key):
        """The design is a deliverable on its own: the rule is frozen, the
        requirement is derived, and the gap is named — with nothing drawn."""
        assert family.report()[key]


class TestTheAllocationRuleVersioning:
    """v2 binds what decides selection; v1 stays recomputable."""

    def test_v1_still_reproduces_its_committed_id(self):
        """The one thing a prospective freeze cannot survive is a committed hash
        quietly meaning something else.

        This caught a real regression: factoring v1's `distribution_identity`
        into a shared helper left a trailing comma, so the helper returned a
        one-element TUPLE and v1's id moved from `ced017a1` to `8d266c70`. The
        shape looked right in every other respect.
        """
        assert family.allocation_rule_id_of(family.allocation_rule_v1()) == \
            family.V1_RULE_ID_AS_COMMITTED

    def test_v2_still_reproduces_its_committed_id(self):
        """Preserved as a JSON snapshot rather than reconstructed in code, after
        reconstructing v1 nearly moved its hash."""
        assert family.allocation_rule_id_of(family.allocation_rule_v2()) == \
            family.V2_RULE_ID_AS_COMMITTED

    def test_v3_is_the_live_rule_and_supersedes_both(self):
        """v2 enumerated three id schemes because the source decision concerned
        three strata. The builder draws all SEVEN, so all seven schemes decide
        which candidate wins -- caught before any row was drawn."""
        rule = family.allocation_rule()
        assert rule["allocation_rule_version"] == 3
        live = family.allocation_rule_id()
        assert live not in (family.V1_RULE_ID_AS_COMMITTED,
                            family.V2_RULE_ID_AS_COMMITTED)
        superseded = {s["version"]: s["id"] for s in rule["_supersedes"]}
        assert superseded == {1: family.V1_RULE_ID_AS_COMMITTED,
                             2: family.V2_RULE_ID_AS_COMMITTED}
        assert "all SEVEN" in superseded_why(rule, 2)
        assert "preserved" in rule["_superseded_rules_are_preserved"]

    def test_the_id_schemes_are_derived_from_the_one_declaration(self):
        from experiments.phase_d_series.identity import ID_SCHEME

        sem = family.allocation_rule()["selection"]["ranking_stable_id_semantics"]
        assert sorted(sem["schemes"]) == sorted(ID_SCHEME)
        assert "fell behind the code" in sem["_schemes_are_derived"]

    def test_the_ranking_identity_semantics_are_bound(self):
        """Changing the id scheme changes the order and therefore the selection,
        so it cannot be deferred to materialization."""
        sel = family.allocation_rule()["selection"]
        sem = sel["ranking_stable_id_semantics"]
        assert "SPLIT-AWARE" in sem["for_new_d_series_rows"]
        assert "d_series_item_id" in sem["for_new_d_series_rows"]
        assert "unchanged" in sem["for_historical_rows"]
        assert "POSITION" in sem["_why_the_historical_scheme_cannot_be_reused"]
        #: and every role's rank domain is in the rule
        assert set(sel["rank_domains"]) == {r[0] for r in family.ROLES}

    def test_all_three_isolation_coordinates_are_bound(self):
        ex = family.allocation_rule()["exclusion"]
        assert len(ex["identity_kinds"]) == 3
        joined = " ".join(ex["identity_kinds"]).lower()
        assert "native/source identity" in joined
        assert "rendered-prompt exact identity" in joined
        assert "problem-content identity" in joined
        assert "all_three_required" in ex
        assert "0 of the 1,708" in ex["all_three_required"]
        #: prior roles and the training corpus by FIRST USER PROBLEM
        reserved = " ".join(ex["reserved_populations_for_problem_content"])
        assert "FIRST USER PROBLEM" in reserved
        assert "prior D-series role" in reserved
        assert "not a repair of the historical function" in ex[
            "historical_chain_is_unmodified"]

    def test_the_frozen_review_is_bound_into_the_rule(self):
        rv = family.allocation_rule()["review"]
        assert rv["provenance"]["retained"] == 43
        assert rv["exclusions"]["code"][0]["native_task_id"] == 602
        assert "RETAINED by decision" in rv["retained_explicitly"]
        assert "excluded nothing by itself" in rv["threshold_role"]

    def test_the_source_policy_is_bound_into_the_rule(self):
        sp = family.allocation_rule()["source_policy"]
        assert "full/train" in sp["code"] and "full/validation" in sp["code"]
        assert "main/train" in sp["gsm8k"]
        assert "TEST splits only" in sp["math_verified"]
        excluded = " ".join(sp["excluded_populations"])
        for name in ("socratic", "sanitized", "MATH train", "second code"):
            assert name in excluded
        assert "DECIDED and IMPLEMENTED" in sp["status"]

    def test_the_rule_no_longer_calls_the_sources_an_open_decision(self):
        rule = family.allocation_rule()
        assert "open maintainer decision" not in rule["what_is_NOT_in_this_rule"]
        assert "DIGESTS" in rule["what_is_NOT_in_this_rule"]

    @pytest.mark.parametrize("mutate", [
        lambda r: r["selection"]["ranking_stable_id_semantics"].__setitem__(
            "for_new_d_series_rows", "historical ids"),
        lambda r: r["exclusion"]["identity_kinds"].pop(),
        lambda r: r["review"]["provenance"].__setitem__("retained", 0),
        lambda r: r["source_policy"].__setitem__("gsm8k", "socratic"),
    ])
    def test_every_bound_section_moves_the_rule_id(self, mutate):
        """If a section can change without moving the hash, it is not bound."""
        rule = family.allocation_rule()
        before = family.allocation_rule_id_of(rule)
        mutate(rule)
        assert family.allocation_rule_id_of(rule) != before


def superseded_why(rule, version: int) -> str:
    return next(s["why"] for s in rule["_supersedes"] if s["version"] == version)

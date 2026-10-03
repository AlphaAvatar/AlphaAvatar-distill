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

REPO = Path(__file__).resolve().parents[2]
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

    def test_both_identity_kinds_are_required(self):
        """Either alone is insufficient — ids miss one question entering the
        pool twice, content hashes miss a paraphrase under one id."""
        chain = family.exclusion_chain()
        assert len(chain["identity_kinds"]) == 2
        kinds = " ".join(chain["identity_kinds"]).lower()
        assert "id" in kinds and "normalized prompt content" in kinds
        assert "insufficient" in chain["both_required"]


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

    def test_the_content_id_is_absent_rather_than_fabricated(self):
        """It binds source pins and realized item digests, neither of which
        exists. A value here would be invented provenance."""
        report = family.report()
        assert report["family_content_id"] is None
        assert "fabricated" in report["_family_content_id_is_null"]

    def test_the_rule_excludes_the_things_it_cannot_know_yet(self):
        """No repository, revision or digest appears in the rule. If one did,
        the rule could not be frozen until the sources were decided — which is
        the decision this round explicitly does not take."""
        rule = family.allocation_rule()
        rendered = json.dumps(rule)
        for forbidden in ("repo_id", "revision", "parquet", "huggingface",
                          "hendrycks", "openai/", "google-research"):
            assert forbidden not in rendered.lower(), forbidden
        #: `sha256` DOES appear — in the order formula, which is the mechanism
        #: and not a pin. The distinction is the point: the rule may describe
        #: how it will hash, and may not name what it will hash over.
        assert "sha256" in rule["selection"]["order"].lower()
        assert "sources" not in rule
        assert "source pins" in rule["what_is_NOT_in_this_rule"]
        assert "unfreezable" in rule["what_is_NOT_in_this_rule"]

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
        assert "None is built" in joined


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

    def test_the_blocker_names_all_three_shortfalls(self):
        blocker = family.report()["blocker"]
        for stratum in ("math_verified", "code", "gsm8k"):
            assert stratum in blocker
        assert "maintainer decision" in blocker


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
        assert doc["_materializes"] == "nothing"
        assert doc["status"].startswith("DESIGNED / NOT MATERIALIZED")

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

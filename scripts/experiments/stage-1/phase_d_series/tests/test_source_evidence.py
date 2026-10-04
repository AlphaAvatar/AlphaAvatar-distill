"""The source-evidence record is derived, and it never calls a row eligible.

These are this experiment family's own tests, not core's: they read the D-series
candidate strata and the committed evidence record.
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

from experiments.phase_d_series import source_evidence as ev  # noqa: E402

RECORD = REPO / ev.RECORD

#: The candidate measurements read the pinned snapshots out of the hub cache and
#: the 76 MB training corpus. A checkout without them cannot answer, and saying
#: so is better than a test that quietly measures nothing.
requires_the_sources = pytest.mark.skipif(
    not (REPO / "artifacts/stage3/corpus_v2/sessions.jsonl").is_file()
    or not (REPO / "artifacts/stage3/c1_confirmation_v1/code.jsonl").is_file(),
    reason=("the recovery corpus and the consumed pools are gitignored "
            "out-of-tree assets; a checkout without them has no overlap to "
            "measure"))


@pytest.fixture(scope="module")
def record():
    if not RECORD.is_file():
        pytest.skip(f"{ev.RECORD} has not been generated")
    return json.loads(RECORD.read_text())


class TestTheRecordClaimsNothingItHasNotMeasured:
    def test_it_authorizes_nothing_and_says_so(self, record):
        assert record["_authorizes"] == "nothing"
        assert "AUTHORIZES NOTHING" in record["_contract"]
        assert "MATERIALIZES NOTHING" in record["_contract"]

    def test_only_the_full_chain_output_is_called_eligible(self, record):
        """The word belongs to ONE place: the complete chain's output.

        Upstream rows and baseline-chain survivors are not eligible counts --
        the baseline chain is not the whole contract, and the D-series owes more
        isolation on top of it. Any other appearance of the word must be saying
        it is not yet known, or is still owed.
        """
        for name, s in record["strata"].items():
            if "eligible_rows" not in s:
                continue
            assert "_this_is_the_eligible_count" in s["eligible_rows"]
            assert "complete live chain" in s["eligible_rows"][
                "_this_is_the_eligible_count"]

        #: everywhere else, scanned by INDEX -- consecutive occurrences are close
        #: enough that a regex window consumed the next match's leading context.
        blob = json.dumps(record)
        at, offences = 0, []
        allowed = ("not ", "owed", "only its output", "this_is_the_eligible_count",
                   "eligible_rows", "output of the complete", "under exact",
                   "first eligible", "would be that stratum's",
                   "eligible under the chain as it is")
        while True:
            i = blob.find("eligible", at)
            if i < 0:
                break
            ctx = blob[max(0, i - 160):i + 80].lower()
            if not any(w in ctx for w in allowed):
                offences.append(blob[max(0, i - 80):i + 60])
            at = i + len("eligible")
        assert not offences, (
            "'eligible' used outside the chain's output and outside a "
            "not-yet-known context: " + " | ".join(offences))

    def test_the_four_levels_are_reported_separately(self, record):
        """upstream -> baseline survivors -> D-series survivors -> eligible.

        Collapsing any two would hide which contract a number was measured
        under, which is how a baseline-only remainder came to be reported as
        capacity in the first version of this record.
        """
        for name, s in record["strata"].items():
            if "upstream_rows" not in s:
                continue
            assert "BEFORE ANY EXCLUSION" in s["upstream_rows"]["_label"]
            base = s["baseline_chain"]
            add = s["d_series_isolation"]
            assert "Not yet eligible" in base["_survivors_label"]
            assert base["survivors"] <= s["upstream_rows"]["candidate_total"]
            assert add["survivors"] <= base["survivors"]
            assert s["eligible_rows"]["count"] == add["survivors"]
            assert (add["removed_beyond_baseline"]
                    == base["survivors"] - add["survivors"])

    def test_the_baseline_chain_names_all_five_populations(self, record):
        """FINAL_PROMOTION was missing from the first version, so this asserts
        the chain's own provenance keys rather than trusting the import."""
        for name, s in record["strata"].items():
            if "baseline_chain" not in s:
                continue
            pops = set(s["baseline_chain"]["populations"])
            assert pops >= {"final_promotion", "recovery_search",
                            "recovery_training", "initializer_state_eval",
                            "operator_calibration"}, pops

    def test_the_screens_disclaim_being_acceptance_criteria(self, record):
        for name, s in record["strata"].items():
            for key in ("near_duplicate_screen", "bare_problem_screen"):
                scr = s.get(key)
                if not scr:
                    continue
                assert "_not_an_acceptance_criterion" in scr, (name, key)


class TestTheFindingsThatBlockANaiveExpansion:
    def test_gsm8k_is_recorded_as_a_renderer_blocker(self, record):
        """The cheapest-looking option carries the identity risk.

        `make_gsm8k` builds its id from the row's POSITION in the file, so the
        train split renders ids that collide with the consumed test split. A
        record that reported only "same repo, 11 items short" would read as the
        easy option.
        """
        assert "gsm8k" in record["renderer_blockers"]
        r = record["strata"]["gsm8k"]["renderer"]
        assert r["id_collisions_with_the_pinned_file"] > 0
        assert r["colliding_ids_that_are_the_same_prompt"] == 0, (
            "if a collision were the same prompt it would be benign; these are "
            "different problems sharing an id")
        assert "BLOCKER" in r

    def test_mbpp_is_not_a_renderer_blocker_but_its_id_lies(self, record):
        code = record["strata"]["code"]
        assert code["renderer"]["id_collisions_with_the_pinned_file"] == 0
        assert "test" in code["renderer"]["id_scheme_note"]

    def test_both_chain_gaps_are_recorded_with_their_demonstrations(self, record):
        """Not as worries: each with the specific thing that proves it."""
        gap = record["chain_gap"]
        #: gap 1 -- a restated problem passes both of the chain's keys
        assert "602" in gap["demonstrated"] and "217" in gap["demonstrated"]
        assert "passes the chain" in gap["demonstrated"]
        assert "decides nothing" in gap["consequence"]
        #: gap 2 -- found only by getting chain parity right
        second = gap["SECOND_GAP_FOUND_BY_CHAIN_PARITY"]
        assert "DISJOINT" in second and "INERT" in second
        assert "caught 0 of the 1,708" in second
        #: and the honest account of why the first version read the other way
        assert "FIRST USER TURN" in gap[
            "_how_the_first_version_of_this_record_got_it_backwards"]
        assert "no historical battery is contaminated" in gap[
            "_why_the_second_gap_never_bit"].lower()

    def test_the_traps_are_named(self, record):
        traps = {t["trap"] for t in record["traps"]}
        assert any("socratic" in t for t in traps)
        assert any("sanitized" in t for t in traps)

    def test_math_verified_is_not_measured_and_says_why(self, record):
        m = record["strata"]["math_verified"]
        assert "NOT MEASURED" in m["status"]
        assert "owed_after_pinning" in m
        assert "upstream_rows" not in m, (
            "an unpinned, uncached source must carry no row count -- quoting a "
            "dataset card's total is the transcription this record avoids")


@requires_the_sources
class TestTheRecordIsDerived:
    def test_it_regenerates_identically(self):
        """A committed evidence record expires silently; this is how a reader
        knows it still describes the sources and the consumed pools."""
        built = json.dumps(ev.report(), indent=1, sort_keys=True) + "\n"
        assert built == RECORD.read_text()

    def test_the_shortfall_comes_from_the_family_not_from_here(self):
        from experiments.phase_d_series.battery_family import requirement

        owned = {n: int(r["short_by"]) for n, r in requirement().items()
                 if int(r["short_by"]) > 0}
        assert ev._shortfall() == owned

    def test_the_stage1_assets_are_subsumed_by_the_corpus_exclusion(self):
        """What makes the exclusion set smaller than it looks.

        The Stage-1 items carry token ids, so they cannot be compared by
        content. If every one traces into `corpus_v2`, excluding the training
        corpus already excludes them. If that ever stops holding, the record's
        remainder is overstated and this fails.
        """
        for asset, got in ev.stage1_provenance().items():
            if got.get("status") == "ABSENT":
                continue
            assert got["traceable_into_corpus_v2"] == got["items_in_these_strata"], (
                f"{asset} holds items in these strata that are NOT from "
                "corpus_v2, so the corpus exclusion no longer subsumes it")

@requires_the_sources
class TestParityWithTheLiveChain:
    """The evidence's exclusion is the BUILDER's exclusion, demonstrated.

    Importing `excluded_identities` and `rank_take` makes drift impossible in
    principle. This proves it in fact: re-deriving the frozen
    `c1_confirmation_v1` battery from the pinned sources, with the imported
    chain and the same C0 digest, must reproduce the committed membership
    exactly. If the assets ever change under the chain, or the chain's reading
    of them changes, this is where it shows.

    Membership, not order: the committed `.jsonl` is written id-sorted while
    `rank_take` returns rank order. Asserting order would pin a serialization
    detail and would have failed while nothing was wrong.
    """

    @pytest.mark.parametrize("stratum", ["gsm8k", "code"])
    def test_the_frozen_c1_battery_rederives_exactly(self, stratum):
        import json as _json
        from battery_render import FROZEN_SOURCES, RENDERERS, rank_take, read_rows
        from build_c1_confirmation_battery import C0_DIGEST, SETS

        ids, hashes, _ = ev.baseline_chain()
        want = SETS[stratum][1]
        repo, revision, rel = FROZEN_SOURCES[stratum]
        rows = ev._indexed(stratum, read_rows(repo, revision, rel))
        rederived = {str(i["id"]) for i in rank_take(
            rows, want, stratum=stratum, base_digest=C0_DIGEST,
            exclude_ids=ids, exclude_hashes=hashes, make=RENDERERS[stratum])}

        committed_path = (REPO / "artifacts/stage3/c1_confirmation_v1"
                          / f"{stratum}.jsonl")
        if not committed_path.is_file():
            pytest.skip("the frozen c1_confirmation_v1 battery is not present")
        committed = {str(_json.loads(l)["id"])
                     for l in committed_path.read_text().splitlines() if l.strip()}

        assert len(rederived) == want
        assert rederived == committed, (
            f"{stratum}: re-deriving the frozen battery with the imported chain "
            f"does not reproduce it. {len(rederived - committed)} extra, "
            f"{len(committed - rederived)} missing. Either the baseline assets "
            "changed under the chain or the evidence is not measuring what the "
            "builder measured.")

    def test_survivors_uses_the_same_predicate(self):
        """`survivors()` must be `rank_take` with the cap lifted, not a copy.

        A hand-written eligibility loop would be a second definition of
        eligible. This checks the function returns every eligible item rather
        than a `want`-limited slice, which is the one way the reuse could still
        be wrong.
        """
        from battery_render import FROZEN_SOURCES, read_rows

        repo, revision, rel = FROZEN_SOURCES["code"]
        rows = read_rows(repo, revision, rel)
        ids, hashes, _ = ev.baseline_chain()
        got = ev.survivors("code", rows, ids, hashes)
        assert 0 < len(got) <= len(rows)
        #: with no exclusions at all, every distinct-prompt row survives
        unbounded = ev.survivors("code", rows, set(), set())
        assert len(unbounded) >= len(got)
        assert len(unbounded) == len({i["prompt_sha256"] for i in unbounded}), (
            "the within-battery dedup is part of the predicate and must apply")


@requires_the_sources
class TestTheTrainingCorpusContentGap:
    def test_the_gap_is_measured_and_recorded(self, record):
        """The finding that inverts the gsm8k recommendation.

        The chain hashes each session's non-assistant messages JOINED, and every
        corpus session carries a system message, so that hash can never equal a
        bare rendered question. For a new upstream source there are no corpus
        `source_id`s either, so the training-corpus protection is inert.
        """
        gap = record["strata"]["gsm8k"]["training_corpus_content_gap"]
        assert gap["the_two_sets_overlap_by"] == 0, (
            "if the joined and first-turn hash sets overlapped, the chain would "
            "catch bare-prompt candidates and this finding would be wrong")
        assert gap["candidates_caught_by_the_chain"] == 0
        assert gap["IN_THE_TRAINING_CORPUS_BUT_NOT_CAUGHT"] > 0
        assert "INERT" in gap["_verdict"]

    def test_it_never_contaminated_the_historical_batteries(self):
        """Why this was latent rather than a past defect.

        The corpus drew gsm8k from `main/train` and every battery from
        `main/test`, so split separation protected them. This asserts that, and
        it is the reason an extension INTO `main/train` is the case where the
        inert hash becomes load-bearing.
        """
        import json as _json
        from aadistill.data.extra_stream import content_sha256
        from battery_render import FROZEN_SOURCES, norm, read_rows

        first_turn = set()
        with (REPO / ev.BASELINE_INPUTS["sessions"]).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                for m in (_json.loads(line).get("messages") or []):
                    if m.get("role") == "user":
                        q = str(m.get("content") or "")
                        if q:
                            first_turn.add(content_sha256(norm(q)))
                        break

        repo, revision, rel = FROZEN_SOURCES["gsm8k"]
        test_rows = read_rows(repo, revision, rel)
        assert not any(content_sha256(norm(r["question"])) in first_turn
                       for r in test_rows), (
            "a gsm8k TEST row is in the training corpus: the historical "
            "batteries are contaminated and this is no longer a latent gap")

        for pool in (*ev.D_SERIES_ADDITIONAL_POOLS, "recovery_search_v2"):
            for r in ev._pool_rows(pool, "gsm8k"):
                assert content_sha256(norm(r["prompt_text"])) not in first_turn, (
                    f"{pool} holds a prompt that is also a training-corpus "
                    "first user turn")


class TestTheProposalIsAProposal:
    """It records intent for a maintainer decision and claims no authority."""

    def test_it_is_labelled_a_proposal_and_authorizes_nothing(self, record):
        prop = record["proposed_source_decision"]
        assert "A PROPOSAL" in prop["_status"]
        assert "NOT a decision" in prop["_status"]
        assert "nothing is pinned or materialized" in prop["_status"].lower()

    def test_gsm8k_carries_both_of_its_blockers(self, record):
        """The renderer AND the training-corpus gap. Either alone stops it, and
        the second was only found by getting chain parity right."""
        blocked = " ".join(record["proposed_source_decision"]["gsm8k"]["BLOCKED_ON"])
        assert "positional renderer" in blocked
        assert "training-corpus content gap" in blocked
        assert "catches 0" in blocked

    def test_gsm8k_count_is_marked_preliminary(self, record):
        g = record["proposed_source_decision"]["gsm8k"]
        assert "preliminary" in g["_the_count_is_preliminary"].lower() or \
            "not known" in g["_the_count_is_preliminary"]

    def test_math_is_a_candidate_to_pin_not_a_source(self, record):
        m = record["proposed_source_decision"]["math_verified"]
        assert "CANDIDATE TO PIN" in m["status"]
        assert "not as a materialized" in m["status"]
        assert "TEST split" in m["preferred_population"]
        assert "not MATH train" in m["preferred_population"]
        #: the adapter is not pretended to be compatible
        adapter = " ".join(m["adapter_requirements"])
        assert "NOT directly compatible" in adapter
        assert "type` -> `subject" in adapter
        assert "NOT imported" in m["scope"]

    def test_the_duplicate_policy_adds_an_exact_key_and_no_threshold(self, record):
        pol = record["proposed_source_decision"]["duplicate_policy_this_round"]
        assert "exact PROBLEM-CONTENT identity" in pol["add"]
        assert "automatic exclusion above a similarity number" in pol["do_not_add"]
        assert "STOP and report" in pol["stop_condition"]

    def test_the_review_list_is_a_surface_not_a_threshold(self, record):
        rl = record["strata"]["code"]["bare_problem_screen"]["review_list"]
        assert rl["n_to_review"] > 0
        assert rl["identical_problem_text"] >= 1
        assert "NOT an equivalence criterion" in rl["_what"]
        assert "BEFORE any D1 outcome" in rl["_when"]
        #: each pair is readable: both problems and both native ids
        for e in rl["pairs"]:
            assert e["candidate_problem"] and e["consumed_problem"]
            assert e["candidate_task_id"] != e["resembles_consumed_task_id"]


@requires_the_sources
class TestTheMathPinningReadiness:
    """The MATH decision is not blocked on a download for the part that isn't."""

    def test_the_answer_derivation_rule_is_verified_not_asserted(self, record):
        """Upstream has `solution` and no `answer`, so the adapter must derive
        the gold. The rule is checked against the EXISTING stratum's own gold
        field, which is what makes it the same correctness semantics rather than
        a new one arriving with a new source."""
        rule = record["math_pinning_readiness"]["answer_derivation_rule"]
        assert rule["rule"] == "aadistill.data.verify.boxed_answer(solution)"
        got, total = rule["reproduces_the_pinned_gold_answer"].split("/")
        assert got == total and int(total) > 0, rule
        assert "VERIFIED" in rule["status"]

    def test_the_distribution_baseline_predates_the_decision(self, record):
        """The owed shift measurement needs something to shift FROM, recorded
        before the candidate is pinned so it cannot be chosen afterwards."""
        base = record["math_pinning_readiness"]["distribution_baseline"]
        assert sum(base["levels"].values()) == int(
            record["math_pinning_readiness"]["parity_baseline"]["rows"])
        assert len(base["subjects"]) >= 5

    def test_it_does_not_presume_the_download(self, record):
        m = record["math_pinning_readiness"]
        assert "maintainer data decision" in m["still_requires_a_download_decision"]
        assert "cannot be derived" in m["still_requires_a_download_decision"]

    def test_the_unique_id_gap_is_named(self, record):
        mapping = record["math_pinning_readiness"]["field_mapping_required"]
        assert mapping["upstream_to_existing"]["type"] == "subject"
        assert "no `unique_id`" in mapping["_unique_id"]
        assert "SEPARATE coordinate" in mapping["_unique_id"]

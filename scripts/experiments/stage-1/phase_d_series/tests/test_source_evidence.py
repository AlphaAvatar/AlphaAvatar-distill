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
from experiments.phase_d_series.identity import (  # noqa: E402
    problem_content_id,
)

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

    def test_exactly_one_authoritative_eligible_count_per_stratum(self, record):
        """ONE numeric eligible count per stratum, at `eligible_rows.count`.

        Not a whitelist of the key NAME: `against_the_family_shortfall` used to
        carry its own numeric `eligible_rows`, and a name-based allowance would
        have accepted it. It read 464/7043 against the authoritative 463/5433,
        with prose saying the problem-content key was still owed — a second
        machine-readable value that went stale the moment the real one moved.

        So the check counts numeric eligible-ish fields by PATH and requires the
        path to be exactly `eligible_rows.count`. A copy anywhere else fails,
        whatever it is called.
        """
        def numeric_eligible_paths(node, path=""):
            found = []
            if isinstance(node, dict):
                for key, value in node.items():
                    here = f"{path}.{key}" if path else key
                    if isinstance(value, (int, float)) and not isinstance(
                            value, bool) and "eligible" in here.lower():
                        found.append(here)
                    found += numeric_eligible_paths(value, here)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    found += numeric_eligible_paths(v, f"{path}[{i}]")
            return found

        for name, s in record["strata"].items():
            if "eligible_rows" not in s:
                continue
            paths = numeric_eligible_paths(s)
            assert paths == ["eligible_rows.count"], (
                f"{name}: expected exactly one authoritative eligible count at "
                f"eligible_rows.count, found {paths}")

            #: and it IS the strengthened output, not a coincidence
            assert s["eligible_rows"]["count"] == s[
                "strengthened_contract"]["survivors"], name
            assert "strengthened chain" in s["eligible_rows"][
                "_this_is_the_eligible_count"], name
            assert "NOT an eligible count" in s[
                "current_exact_chain_survivors"]["_what_this_is_NOT"], name

    def test_the_shortfall_block_references_rather_than_copies(self, record):
        """`headroom` must be derived from the authoritative count."""
        for name, s in record["strata"].items():
            block = s.get("against_the_family_shortfall")
            if not block:
                continue
            assert block["eligible_ref"] == "eligible_rows.count", name
            assert "eligible_rows" not in block or not isinstance(
                block.get("eligible_rows"), (int, float)), name
            assert (block["headroom"]
                    == s["eligible_rows"]["count"] - block["six_role_shortfall"]), name

    def test_the_record_describes_current_state_not_a_proposal(self, record):
        """The decision was made and implemented; a live object that still calls
        it proposed, or still owes what has been done, is describing a state the
        repository is no longer in."""
        assert "proposed_source_decision" not in record
        decided = record["source_decision"]
        assert "DECIDED" in decided["_status"] and "IMPLEMENTED" in decided["_status"]

        owed = " ".join(record["still_owed_before_any_materialization"]).lower()
        for closed in ("source decision", "math pin", "problem-content key",
                       "exclusion chain, run against the pinned candidate"):
            assert closed not in owed, f"{closed!r} is listed as owed but is done"
        assert "renderer" in owed and "materialization decision" in owed
        assert "_what_is_no_longer_owed" in record

        #: the pinning block must not still ask for a download decision
        assert "still_requires_a_download_decision" not in record[
            "math_pinning_readiness"]

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
            upstream = s["upstream_rows"].get("candidate_total") or \
                s["upstream_rows"]["total"]
            assert base["survivors"] <= upstream
            assert add["survivors"] <= base["survivors"]
            cur = s["current_exact_chain_survivors"]["count"]
            assert cur == add["survivors"]
            assert (add["removed_beyond_baseline"]
                    == base["survivors"] - add["survivors"])
            #: and the strengthened level is the LAST one, never above the one
            #: before it -- a coordinate that admits more is not an isolation
            assert s["eligible_rows"]["count"] <= cur

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

    def test_math_verified_is_now_pinned_and_measured(self, record):
        """It was "NOT MEASURED" because it was not pinned. The maintainer
        source decision authorized pinning and fetching it, so now the record
        must carry real counts -- and they must come from the pin, not a card."""
        m = record["strata"]["math_verified"]
        assert "upstream_rows" in m, "a pinned source owes measured counts"
        assert m["upstream_rows"]["total"] > 0
        assert len(m["pin"]["revision"]) == 40
        assert sum(m["upstream_rows"]["per_subject"].values()) == \
            m["upstream_rows"]["total"]


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


class TestTheDecisionIsRecordedAsDecided:
    """It WAS a proposal; it is now made and implemented.

    This class replaces `TestTheProposalIsAProposal`, which asserted the
    pre-decision state -- that MATH was "a candidate to pin, not a source", that
    the gsm8k count was "preliminary" because the proposal had not been accepted.
    Those assertions were correct and are now false, and a test that pins a
    superseded state is a test that must be edited to tell the truth.
    """

    def test_the_sources_are_recorded_as_decided(self, record):
        d = record["source_decision"]
        assert "mbpp" in d["code"]["source"]
        assert "main/train" in d["gsm8k"]["source"]
        assert "PINNED AND FETCHED" in d["math_verified"]["source"]

    def test_gsm8k_still_carries_its_remaining_blocker(self, record):
        """Decided is not usable: the renderer is still missing, and the
        distinction between capacity and rendered membership is the point."""
        blocker = record["source_decision"]["gsm8k"]["REMAINING_BLOCKER"]
        assert "RENDERER" in blocker
        assert "not be reused" in blocker
        assert "Capacity is established" in blocker
        #: and the trap list says the same thing where a reader meets the number
        traps = {t["trap"]: t["why_not"] for t in record["traps"]}
        key = next(k for k in traps if "rendered membership" in k)
        assert "conservative capacity evidence" in traps[key]
        assert "no new source decision" in traps[key]

    def test_math_is_recorded_as_a_pinned_source_now(self, record):
        m = record["source_decision"]["math_verified"]
        assert "PINNED AND FETCHED" in m["source"]
        assert "NOT used" in m["train_split"]
        assert "level 'Level N'->N" in m["adapter"]
        assert "not imported" in m["scope"]

    def test_the_duplicate_policy_is_an_exact_key_and_a_frozen_review(self, record):
        pol = record["source_decision"]["duplicate_policy"]
        assert "exact canonical problem-content identity" in pol["added"]
        assert "automatic similarity cutoff" in pol["not_added"]
        assert "602" in pol["frozen_review"] and "43" in pol["frozen_review"]
        assert "before any D1 outcome" in pol["frozen_review"]


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

    def test_the_readiness_block_is_now_the_parity_baseline(self, record):
        """It existed to show the pinning decision was not blocked on a download
        for the part that did not need one. The download has happened, so what it
        keeps is the baseline the measured counts were checked against."""
        m = record["math_pinning_readiness"]
        assert "still_requires_a_download_decision" not in m
        assert "MADE and EXECUTED" in m["_download_decision"]
        assert m["answer_derivation_rule"]["status"].startswith("VERIFIED")

    def test_the_unique_id_gap_is_named(self, record):
        mapping = record["math_pinning_readiness"]["field_mapping_required"]
        assert mapping["upstream_to_existing"]["type"] == "subject"
        assert "no `unique_id`" in mapping["_unique_id"]
        assert "SEPARATE coordinate" in mapping["_unique_id"]


class TestTheTerminologyIsNotSelfContradictory:
    """The record said "no row is called eligible" beside `eligible_rows = 464`."""

    def test_the_contract_does_not_deny_what_the_record_reports(self, record):
        c = record["_contract"]
        assert "No row here is called eligible" not in c
        assert "has not been run against any candidate" not in c
        assert "STRENGTHENED" in c

    def test_the_current_contract_level_is_not_called_eligible(self, record):
        for name, s in record["strata"].items():
            cur = s.get("current_exact_chain_survivors")
            if not cur:
                continue
            assert "NOT an eligible count" in cur["_what_this_is_NOT"]
            #: and it is >= the eligible count, never equal by accident of naming
            assert cur["count"] >= s["eligible_rows"]["count"]

    def test_eligible_means_the_strengthened_output(self, record):
        for name, s in record["strata"].items():
            if "strengthened_contract" not in s:
                continue
            assert s["eligible_rows"]["count"] == s["strengthened_contract"]["survivors"]
            assert "strengthened chain" in s["eligible_rows"][
                "_this_is_the_eligible_count"]


@requires_the_sources
class TestTheStrengthenedContract:
    def test_it_does_not_change_the_historical_chain(self):
        """The whole reason it is a layer: frozen battery membership must stay
        reproducible from the builder's own function."""
        import inspect

        from build_c1_confirmation_battery import excluded_identities

        src = inspect.getsource(excluded_identities)
        assert "problem_content" not in src, (
            "the historical chain has grown a problem-content key; frozen "
            "C1/C2/C3 membership is no longer reproducible from it")

    def test_no_recovery_training_row_survives_for_gsm8k(self, record):
        """The blocker the maintainer named. The historical chain catches 0 of
        the 1,708; the strengthened contract must catch all of them."""
        st = record["strata"]["gsm8k"]["strengthened_contract"]
        gap = record["strata"]["gsm8k"]["training_corpus_content_gap"]
        assert gap["IN_THE_TRAINING_CORPUS_BUT_NOT_CAUGHT"] > 0, (
            "the gap is the premise; if it closed, this stratum's story changed")
        #: recompute the survivor set and check it directly
        import hashlib as _h
        import json as _j

        from battery_render import FROZEN_SOURCES, RENDERERS, norm, read_rows

        train = set()
        with (REPO / ev.BASELINE_INPUTS["sessions"]).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                for m in (_j.loads(line).get("messages") or []):
                    if m.get("role") == "user":
                        q = str(m.get("content") or "")
                        if q:
                            train.add(_h.sha256(norm(q).encode()).hexdigest())
                        break
        repo, revision, _ = FROZEN_SOURCES["gsm8k"]
        rows = read_rows(repo, revision, "main/train-00000-of-00001.parquet")
        base_ids, base_hashes, _ = ev.baseline_chain()
        add_ids, add_hashes, _ = ev.d_series_additional("gsm8k")
        surv = ev.survivors("gsm8k", rows, base_ids | add_ids,
                            base_hashes | add_hashes)
        src = {}
        for i, r in enumerate(rows):
            item = RENDERERS["gsm8k"](dict(r, _index=i))
            if item:
                src[str(item["id"])] = dict(r, _index=i)
        reserved = ev.reserved_problem_content("gsm8k")["ids"]
        seen, leaked = set(), 0
        for item in surv:
            row = src[str(item["id"])]
            content = problem_content_id("gsm8k", row)
            if content in reserved or content in seen:
                continue
            seen.add(content)
            if _h.sha256(norm(row["question"]).encode()).hexdigest() in train:
                leaked += 1
        assert leaked == 0, (
            f"{leaked} recovery-training rows survive the strengthened contract")
        assert len(seen) == st["survivors"]

    def test_every_reserved_problem_is_recoverable(self, record):
        """An unrecoverable reserved problem is a hole in the isolation, and
        reporting one as zero would be the dangerous direction."""
        for name, s in record["strata"].items():
            st = s.get("strengthened_contract")
            if not st:
                continue
            assert st["reserved_unrecoverable"] == 0, (
                f"{name}: {st['reserved_unrecoverable']} reserved problems "
                "could not be content-hashed, so the contract is blind to them")

    def test_the_frozen_review_is_data_not_a_threshold(self, record):
        from experiments.phase_d_series.identity import REVIEW_PROVENANCE

        rv = record["strata"]["code"]["strengthened_contract"]["frozen_review"]
        assert rv["n_excluded"] == 1
        assert rv["excluded"][0]["native_task_id"] == 602
        assert rv["excluded"][0]["duplicate_of_consumed_task_id"] == 217
        assert rv["provenance"]["retained"] == 43
        assert rv["provenance"]["frozen_before_any_d1_outcome"] is True
        assert REVIEW_PROVENANCE["trigger_threshold"] == 0.8


class TestTheDSeriesIdentityCoordinates:
    """Three coordinates, and they must not be conflated."""

    def test_the_item_id_is_split_aware_for_every_source(self):
        from experiments.phase_d_series.identity import d_series_item_id

        assert d_series_item_id("code", "full", "train", {"task_id": 602}) == \
            "mbpp-full-train-602"
        assert d_series_item_id("gsm8k", "main", "train", {}, index=0) == \
            "gsm8k-main-train-00000"
        #: and a non-test MBPP row never gets a `mbpp-test-` id
        assert not d_series_item_id(
            "code", "full", "train", {"task_id": 602}).startswith("mbpp-test-")

    def test_gsm8k_without_an_index_is_refused(self):
        from experiments.phase_d_series.identity import d_series_item_id

        with pytest.raises(ValueError, match="no native key"):
            d_series_item_id("gsm8k", "main", "train", {})

    def test_problem_content_is_independent_of_the_rendering(self):
        """The point of the coordinate: two different renderings of one problem
        share a content id, which is what the historical chain could not see."""
        a = {"text": "Write a function to add two numbers."}
        b = {"text": "write  a FUNCTION to add two numbers. "}
        assert problem_content_id("code", a) == problem_content_id("code", b)

    def test_an_empty_problem_is_refused_not_hashed(self):
        with pytest.raises(ValueError, match="no 'text'|carries no"):
            problem_content_id("code", {"text": "   "})

    def test_an_undeclared_source_is_refused(self):
        with pytest.raises(KeyError, match="no problem payload"):
            problem_content_id("not_a_source", {"text": "x"})

    def test_hashing_a_wrapped_prompt_as_content_is_refused(self):
        """MBPP's rendered prompt wraps the problem, so hashing it would make
        every wrapped item look like a distinct problem."""
        from experiments.phase_d_series.identity import (
            problem_content_id_from_prompt,
        )

        with pytest.raises(ValueError, match="wraps the problem"):
            problem_content_id_from_prompt("code", "instruction + problem")
        #: and it is allowed where the rendering IS the problem
        assert problem_content_id_from_prompt("gsm8k", "a question")

    def test_the_declared_rendering_property_matches_the_renderers(self):
        """Declared, then verified against the real renderers -- the first
        version assumed only a native-key route and reported 430 reserved
        problems as unrecoverable when their text was in `prompt_text`."""
        from battery_render import FROZEN_SOURCES, RENDERERS, norm, read_rows
        from experiments.phase_d_series.identity import (
            PROBLEM_FIELD,
            RENDERED_PROMPT_IS_THE_PROBLEM,
        )

        for group, declared in RENDERED_PROMPT_IS_THE_PROBLEM.items():
            if group not in FROZEN_SOURCES:
                continue
            repo, revision, rel = FROZEN_SOURCES[group]
            rows = read_rows(repo, revision, rel)[:50]
            if group == "gsm8k":
                rows = [dict(r, _index=i) for i, r in enumerate(rows)]
            make, field = RENDERERS[group], PROBLEM_FIELD[group]
            same = all(norm(make(r)["prompt_text"]) == norm(str(r[field]))
                       for r in rows)
            assert same == declared, (
                f"{group}: RENDERED_PROMPT_IS_THE_PROBLEM says {declared} but "
                f"the renderer says {same}")


@requires_the_sources
class TestThePinnedMathSource:
    def test_the_pin_is_an_immutable_revision_with_a_licence(self, record):
        pin = record["strata"]["math_verified"]["pin"]
        assert pin["repo_id"] == "EleutherAI/hendrycks_math"
        assert len(pin["revision"]) == 40, "a branch name is not a pin"
        assert pin["licence"] == "mit"
        assert "TEST ONLY" in pin["splits_fetched"]

    def test_every_pinned_file_has_a_sha256(self, record):
        files = record["strata"]["math_verified"]["pin"]["files"]
        assert len(files) == 7, "seven configs, test split each"
        for rel, got in files.items():
            assert rel.endswith("test-00000-of-00001.parquet"), rel
            assert len(got["sha256"]) == 64, rel
            assert got["size_bytes"] > 0

    def test_no_train_file_is_pinned(self, record):
        files = record["strata"]["math_verified"]["pin"]["files"]
        assert not any("train" in rel for rel in files), (
            "MATH train must not be used for this behavioural stratum")

    def test_the_adapter_parity_holds(self, record):
        par = record["strata"]["math_verified"]["adapter_parity"]
        assert par["status"] == "PARITY HOLDS"
        got, total = par["reproduces_pinned_gold_on_the_pinned_file"].split("/")
        assert got == total
        agreed, shared = par["gold_agrees_on_shared_problems"].split("/")
        assert agreed == shared and int(shared) > 0

    def test_the_level_mapping_is_applied_not_passed_through(self):
        """Upstream emits `"Level 3"`; the frozen stratum stores `3`."""
        from experiments.phase_d_series.math_source import level_to_int

        assert level_to_int("Level 3") == 3
        assert level_to_int(4) == 4
        with pytest.raises(ValueError, match="cannot read a level"):
            level_to_int("unknown")

    def test_a_row_with_no_boxed_answer_is_refused(self):
        from experiments.phase_d_series.math_source import adapt

        with pytest.raises(ValueError, match="no boxed answer"):
            adapt({"problem": "p", "level": "Level 1", "type": "Algebra",
                   "solution": "no box here"}, config="algebra", index=0)

    def test_a_config_type_disagreement_is_refused(self):
        from experiments.phase_d_series.math_source import adapt

        with pytest.raises(ValueError, match="mapping and the data disagree"):
            adapt({"problem": "p", "level": "Level 1", "type": "Geometry",
                   "solution": r"$\boxed{2}$"}, config="algebra", index=0)

    def test_the_distribution_shift_is_recorded_against_the_frozen_baseline(
            self, record):
        shift = record["strata"]["math_verified"]["distribution_shift"]
        for axis in ("subject", "level"):
            assert set(shift[axis]["candidate"]) == set(
                shift[axis]["frozen_math500"]), axis
            assert abs(sum(shift[axis]["candidate"].values()) - 1.0) < 0.01

    def test_it_is_declared_a_new_population(self, record):
        scope = record["strata"]["math_verified"]["scope"]
        assert "NEW D-series behavioural population" in scope
        assert "NOT imported" in scope

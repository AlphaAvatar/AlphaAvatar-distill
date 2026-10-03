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

    def test_no_count_is_called_eligible(self, record):
        """The word the maintainer reserved for the exclusion chain's output.

        A row is eligible only after the chain has run. Until then every number
        is an upstream row count or a measured overlap, and the record must not
        borrow the stronger word for either.
        """
        blob = json.dumps(record)
        #: Scanned by INDEX, not `re.finditer`. Consecutive occurrences are
        #: close enough together that a regex window consumed the next match's
        #: leading context, and the qualifying words sit before the word.
        at, offences = 0, []
        while True:
            i = blob.find("eligible", at)
            if i < 0:
                break
            ctx = blob[max(0, i - 120):i + 60].lower()
            if not any(w in ctx for w in (
                    "not ", "owed", "first eligible", "output of the exclusion",
                    "only its output")):
                offences.append(blob[max(0, i - 80):i + 60])
            at = i + len("eligible")
        assert not offences, (
            "'eligible' used as a claim rather than as something still owed: "
            + " | ".join(offences))

    def test_every_measured_count_is_labelled_as_what_it_is(self, record):
        for name, s in record["strata"].items():
            if "upstream_rows" not in s:
                continue
            assert "BEFORE ANY EXCLUSION" in s["upstream_rows"]["_label"]
            assert "Not an eligible count" in s["exclusion_measurement"]["_label"]

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

    def test_the_chain_gap_is_recorded_with_its_demonstration(self, record):
        """Not as a worry: with the specific pair that proves it."""
        gap = record["chain_gap"]
        assert "602" in gap["demonstrated"] and "217" in gap["demonstrated"]
        assert "passes the chain" in gap["demonstrated"]
        assert "maintainer decision" in gap["consequence"]

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

"""The snapshot must not contradict itself.

`current_state.json` said both things at once. `phase_c.c1.status` read "TEN
LABELS, NINE PAID / REPLAY MEASURED 2-of-2 PASS" while `phases.phase_c1`, four
keys away, still read "NINE LABELS, EIGHT PAID / NEVER MEASURED" — the
pre-attempt-9 wording, left behind when the attempt-9 closeout updated the
detailed block and not the summary map. A reader who opened the file at the
wrong key would have concluded that C1 has never measured anything.

Two copies of a fact are two chances to be wrong, so the repair was mostly
deletion: the summary map now points at the owner instead of restating it. What
remains is checked here **by scanning every string in the file** rather than by
pinning one blessed key. Pinning a key is what the old arrangement effectively
did — `phase_c.c1` was right the whole time. A rule that only reads the key I
happen to trust cannot see the copy that drifted, so these tests read all of
them and require the claims to agree.

Historical records are deliberately out of scope. The attempt-8 and attempt-9
grants say "Eight attempt labels" and "Nine attempt labels" and were correct
when written; `logs/autoinit_c1_attempt9/outcome.json` likewise. Those are
sealed evidence, not live state.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "logs/current_state.json"
STATE = REPO / "logs/STATE.md"

WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11}


def strings(obj, path=""):
    """Every string in the snapshot, with the key path that holds it."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def count_before(text: str, noun: str) -> list[int]:
    """Numbers claimed immediately before `noun`, as digits or as words."""
    out = []
    for m in re.finditer(rf"\b([a-z]+|\d+)\s+(?:\w+\s+)?{noun}\b", text, re.I):
        tok = m.group(1).lower()
        if tok.isdigit():
            out.append(int(tok))
        elif tok in WORD:
            out.append(WORD[tok])
    return out


# --- the counts agree with each other ---------------------------------------

class TestTheAttemptCountIsOneNumber:
    #: Twelve labels (1, 2, 3, 3R, 4-11) and eleven paid since 2026-09-11, when
    #: attempts 10 and 11 were CLOSED OUT and booked. The counts move with the
    #: LEDGER, not with the appearance of a pod: while attempt 10 was running I
    #: briefly wrote "eleven labels, ten paid" here and reverted it, because a
    #: running attempt is not a booked one and the count is one number.
    LABELS, PAID = 12, 11

    def test_every_label_claim_agrees(self):
        """The exact drift: one key said ten labels and another said nine."""
        bad = {p: v for p, v in strings(snapshot())
               for n in count_before(v, r"labels?")
               if n != self.LABELS}
        assert not bad, (
            f"C1 has {self.LABELS} attempt labels (1, 2, 3, 3R, 4-11). "
            "Disagreeing:\n" + "\n".join(f"  {p}: {v}" for p, v in bad.items()))

    def test_every_paid_claim_agrees(self):
        bad = {p: v for p, v in strings(snapshot())
               for n in count_before(v, r"paid")
               if n != self.PAID}
        assert not bad, (
            f"{self.PAID} attempts were paid; attempt 3 created no resource. "
            "Disagreeing:\n" + "\n".join(f"  {p}: {v}" for p, v in bad.items()))

    def test_the_labels_and_the_paid_count_differ_by_exactly_the_free_one(self):
        """The labels and the paid count are only coherent because exactly one
        label spent nothing. If that stops being true the two numbers stop being
        both right, and this catches it in the ledger rather than the snapshot."""
        ledger = (REPO / "logs/BUDGET_LEDGER.md").read_text()
        free = re.findall(r"C1 attempt 3[^|]*\|\s*\$0\.0000", ledger)
        assert self.LABELS - self.PAID == 1, (
            'more than one free label would need more than one explanation')
        assert free, ("nothing in BUDGET_LEDGER.md still records a C1 label "
                      "that cost $0.0000, so 'ten labels, nine paid' has lost "
                      "its explanation")


# --- the measurement state agrees -------------------------------------------

class TestReplayIsMeasuredEverywhere:
    # The stale key read "NINE LABELS, EIGHT PAID / NEVER MEASURED" and never
    # used the word "replay", so a pattern requiring "replay" nearby missed the
    # one string it existed to catch. Inside this file a bare "never measured"
    # is unconditionally wrong: the only thing C1 has measured is the replay,
    # and an unqualified negative can only be denying it. "UNMEASURED" stays
    # legal because treatment and endpoint genuinely are.
    NEGATIVE = re.compile(
        r"\bnever (been )?measured\b|\bno completed replay\b|"
        r"replay[^.]{0,40}\b(not measured|unmeasured)\b", re.I)

    def test_no_key_says_replay_is_unmeasured(self):
        """Attempt 9 passed both frozen gates. The file may not still say
        otherwise anywhere."""
        bad = {p: v for p, v in strings(snapshot()) if self.NEGATIVE.search(v)}
        assert not bad, (
            "replay is MEASURED (2/2 PASS). Contradicting:\n"
            + "\n".join(f"  {p}: {v}" for p, v in bad.items()))

    def test_the_measurement_state_is_stated_once_and_in_full(self):
        m = snapshot()["phase_c"]["c1"]["measured"]
        assert re.search(r"replay:\s*MEASURED", m, re.I), m
        assert re.search(r"2/2 PASS", m), m
        for unmeasured in ("treatment", "endpoint"):
            assert unmeasured in m.lower(), f"{unmeasured} not addressed: {m}"
        assert "UNMEASURED" in m, m

    def test_treatment_and_endpoint_are_not_claimed_measured(self):
        bad = {p: v for p, v in strings(snapshot())
               if re.search(r"(treatment|endpoint)[^.]{0,30}\bMEASURED\b", v)
               and not re.search(r"(treatment|endpoint)[^.]{0,30}\bUNMEASURED",
                                 v)}
        assert not bad, f"treatment/endpoint claimed measured: {bad}"


# --- the required facts are actually present --------------------------------

class TestTheSnapshotStatesTheRequiredFacts:
    @pytest.mark.parametrize("pattern,fact", [
        (r"\bNO DECISION\b", "attempt 9 is NO DECISION"),
        (r"pre-treatment infrastructure abort", "why it is NO DECISION"),
        # Until 2026-09-10 this required "LOGICAL / CPU-STRUCTURAL EVIDENCE
        # ONLY", which was the repair's standing while no accelerator had ever
        # executed it. A real CUDA device has now observed it, so the required
        # fact is the confirmation and the SHA it is bound to -- an
        # unattributed "confirmed" would be the weaker claim.
        (r"CONFIRMED ON REAL CUDA", "the stage-F repair's evidentiary standing"),
        (r"7027a8f4", "the execution SHA that confirmation is bound to"),
        (r"ENGINEERING EVIDENCE ONLY",
         "that the CUDA validation is not a C1 result"),
    ])
    def test_the_fact_appears(self, pattern, fact):
        blob = "\n".join(v for _, v in strings(snapshot()))
        assert re.search(pattern, blob, re.I), f"the snapshot does not state {fact}"

    def test_the_migration_is_stated_as_engineering_not_as_a_c1_result(self):
        """Asked of the migration's OWN field, not of the whole snapshot.

        This used to require the literal `CURRENT ENGINEERING ACTIVITY` anywhere
        in the document. That phrase stopped being true when Milestone A merged,
        and a blob-wide search would in any case have been satisfied by the CUDA
        validation's own "not a C1 result" — a different subject making the same
        disclaimer. The enduring fact is about the migration, so it is checked
        where the migration is described, and it holds whether the migration is
        running, merged or abandoned.
        """
        status = snapshot()["architecture_migration"]["status"]
        assert re.search(r"NOT a C1 result", status, re.I), status
        assert re.search(r"no ATTENTION evidence", status, re.I), status
        assert re.search(r"authorizes nothing", status, re.I), status

    @pytest.mark.parametrize("pattern,why", [
        (r"CPU-STRUCTURAL EVIDENCE ONLY",
         "the repair is no longer CPU-only evidence"),
        (r"NOT AUTHORIZED[^.]{0,40}(GPU|CUDA)|GPU[^.]{0,30}NOT authorized",
         "the GPU validation was authorized, run and closed"),
        (r"\$?267\.8598\b", "that cumulative is superseded by $267.8998"),
        (r"\$?15\.9002\b", "that remaining figure is superseded by $15.8602"),
    ])
    def test_the_stale_claim_is_gone(self, pattern, why):
        """The other half. A snapshot that states the new fact while still
        carrying the old one contradicts itself, and a reader has no way to
        tell which line is current."""
        bad = {p: v for p, v in strings(snapshot()) if re.search(pattern, v, re.I)}
        assert not bad, f"{why}; still present in:\n" + "\n".join(
            f"  {p}: {v}" for p, v in bad.items())

    def test_the_engineering_campaign_is_closed(self):
        cv = snapshot()["cuda_engineering_validation"]
        assert cv["campaign_cost_usd"] == 0.04
        assert "CLOSED" in cv["campaign"]
        assert cv["authorizes"] == "nothing"
        assert (REPO / cv["amendment"].split()[0]).is_file(), (
            "the snapshot names an interpretation amendment that does not exist")

    def test_an_approved_package_is_not_an_issued_authorization(self):
        """The distinction the whole launch contract rests on.

        This asserted `authorized.any is False` outright, which was the same
        sentence as "no package has been approved" only while none had. On
        2026-09-11 one was, and the invariant worth protecting is not that
        nothing is approved — it is that **an approval is not a launch**: a
        package permits a bounded number of one-use chains, and each chain still
        needs a grant, a launch-bound sweep, an issued authorization and a
        staged bundle before anything can be created.

        So the shape is pinned in whichever direction is true, and the attempt
        counter is required to exist and stay inside the approved bound — a
        package whose usage is untracked is a package without a limit.
        """
        s = snapshot()
        a = s["authorized"]
        if not a["any"]:
            assert s["prepared_launch"]["any"] is False
            assert s["running"]["paid_compute"] is False
            return

        assert a.get("package_id"), (
            "the snapshot claims an authorization exists but names no package")
        used, cap = a.get("formal_attempts_used"), a.get("formal_attempts_max")
        assert isinstance(used, int) and isinstance(cap, int), (
            "an approved package must count its attempts; an uncounted package "
            "has no limit")
        assert 0 <= used <= cap, f"{used} of {cap} attempts used"
        #: A package approval must never be written as though it were the
        #: issuance. If a bundle is staged or a pod is billing, that is a
        #: separate claim the snapshot has to make explicitly elsewhere.
        assert "approval is not a launch" in a["note"].lower() or \
            "not a launch" in a["note"].lower(), (
                "the note does not distinguish approval from issuance")

    def test_the_migration_authorizes_nothing_and_is_not_a_result(self):
        status = snapshot()["architecture_migration"]["status"].lower()
        assert "not a c1 result" in status, status
        assert "authorizes nothing" in status, status

    def test_the_migration_record_it_names_exists(self):
        rec = REPO / snapshot()["architecture_migration"]["record"]
        assert rec.exists(), f"{rec} does not exist"
        assert (rec / "source-relocation.json").is_file(), (
            "the migration record names no source-relocation.json")

    def test_the_stage_f_repair_is_not_called_validated(self):
        """It has not run on a GPU. The snapshot must not imply it has."""
        blob = "\n".join(v for _, v in strings(snapshot()))
        assert not re.search(
            r"stage[- ]F[^.]{0,60}\b(validated|verified on (a )?gpu|"
            r"observed on (a )?gpu|confirmed on cuda)", blob, re.I), blob


# --- the human view agrees with the machine view ----------------------------

class TestStateMdAgrees:
    """STATE.md is the canonical handoff; a repair only in the JSON is invisible
    to the person reading the prose."""

    def live_region(self) -> str:
        """STATE.md down to the first superseded section, with quoted literals
        removed.

        Two exclusions, and both are load-bearing. Everything below the first
        `# Superseded` heading is history, kept as written, and is not a claim
        about now. And an inline-code span is a CITATION: the entry describing
        this very repair has to be able to write ``"NINE LABELS, EIGHT PAID /
        NEVER MEASURED"`` to say what the stale key said. A gate that cannot
        tell a quotation from an assertion forces the documentation to describe
        its own defects vaguely, which is worse than the defect.

        Prose outside backticks is still an assertion, and
        `test_an_unquoted_claim_is_still_caught` holds that line.
        """
        text = STATE.read_text()
        cut = text.find("\n# Superseded")
        live = text[:cut] if cut > 0 else text
        return re.sub(r"`[^`]*`", "`", live)

    def test_it_does_not_claim_nine_labels_or_eight_paid(self):
        live = self.live_region()
        assert not count_before(live, r"attempt labels") or \
            set(count_before(live, r"attempt labels")) == {10}, live[:400]
        bad = [n for n in count_before(live, r"paid") if n != 9]
        assert not bad, f"STATE.md claims {bad} paid attempts"

    def test_it_does_not_say_c1_was_never_measured(self):
        assert not re.search(r"NEVER MEASURED", self.live_region()), (
            "STATE.md's live region still says C1 was never measured; attempt "
            "9 passed both replay gates")

    def test_it_points_at_the_migration_record_rather_than_restating_it(self):
        assert "migrations/initialization-core/v1" in self.live_region()

    def test_the_stale_section_is_marked_rather_than_deleted(self):
        """The attempt-8-era section is history and stays; what it must not do
        is present itself as current."""
        text = STATE.read_text()
        assert "# Superseded: the current state as of attempt 8" in text
        heads = re.findall(r"^# Current state", text, re.M)
        assert len(heads) == 1, (
            f"{len(heads)} sections claim to be the current state")

    def test_an_unquoted_claim_is_still_caught(self, monkeypatch, tmp_path):
        """The backtick exclusion must not become a way to smuggle a claim.

        Prose is an assertion; a code span is a quotation. Only the second is
        exempt, and this pins the difference in both directions.
        """
        import test_current_state_consistency as mod

        quoted = tmp_path / "quoted.md"
        quoted.write_text("The stale key read `NEVER MEASURED` and was wrong.\n")
        plain = tmp_path / "plain.md"
        plain.write_text("Phase C1 is NEVER MEASURED.\n")

        monkeypatch.setattr(mod, "STATE", quoted)
        self.test_it_does_not_say_c1_was_never_measured()   # the citation passes

        monkeypatch.setattr(mod, "STATE", plain)
        with pytest.raises(AssertionError):
            self.test_it_does_not_say_c1_was_never_measured()

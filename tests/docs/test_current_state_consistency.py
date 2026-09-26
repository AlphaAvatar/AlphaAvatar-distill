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
when written; `logs/stages/stage-1/phase_c1/runs/attempt9/outcome.json` likewise. Those are
sealed evidence, not live state.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "logs/state/current.json"
STATE = REPO / "logs/state/current.md"

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
    #: Thirteen labels (1, 2, 3, 3R, 4-12) and eleven paid since 2026-09-11,
    #: when attempts 10, 11 and 12 were CLOSED OUT and booked. The counts move
    #: with the LEDGER, not with the appearance of a pod: while attempt 10 was
    #: running I briefly wrote "eleven labels, ten paid" here and reverted it,
    #: because a running attempt is not a booked one and the count is one
    #: number. Attempt 12 is a label that cost nothing -- the provider refused
    #: the create -- so it moves LABELS and not PAID.
    LABELS, PAID = 13, 11

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

    #: Every label that created no provider resource and therefore cost
    #: nothing. The gap between LABELS and PAID is exactly these, and each one
    #: has to be findable in the ledger at $0.0000 -- the point is that the gap
    #: is EXPLAINED, not that it happens to be some size.
    FREE_LABELS = ("3", "12")

    def test_the_labels_and_the_paid_count_differ_by_exactly_the_free_ones(self):
        """The labels and the paid count are only coherent because some labels
        spent nothing. This checks the arithmetic against the LEDGER rather than
        against the snapshot that makes the claim.

        It asserted a difference of exactly ONE, which was the same sentence as
        "the gap is explained" only while there was one free label. Attempt 12
        was refused at $0 and made a second. Requiring one would have forced the
        true count out of the documentation to keep a test green.
        """
        ledger = (REPO / "logs/budget/ledger.md").read_text()
        missing = [n for n in self.FREE_LABELS
                   if not re.search(rf"C1 attempt {n}[^|]*\|\s*`?\$0\.0000",
                                    ledger)]
        assert not missing, (
            f"BUDGET_LEDGER.md records no $0.0000 C1 label for {missing}, so "
            f"'{self.LABELS} labels, {self.PAID} paid' has lost its explanation")
        assert self.LABELS - self.PAID == len(self.FREE_LABELS), (
            f"{self.LABELS} - {self.PAID} = {self.LABELS - self.PAID} free "
            f"labels, but {len(self.FREE_LABELS)} are named and explained")


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
        for axis in ("treatment", "endpoint"):
            assert axis in m.lower(), f"{axis} not addressed: {m}"

    def test_a_measured_endpoint_cites_the_record_that_measured_it(self):
        """Until 2026-09-14 this required the word UNMEASURED, because nothing
        had measured the endpoint and a claim otherwise could only be false.

        Attempt 18 measured it. The protection that still means something is not
        "never say measured" — it is that a measured claim must be checkable:
        the snapshot must point at the decision record rather than paraphrase
        it, and the record must exist and carry a verdict the frozen rule
        emits.
        """
        m = snapshot()["phase_c"]["c1"]["measured"]
        if "UNMEASURED" in m:
            return                              # nothing measured, nothing owed

        assert "c1_decision.json" in m, (
            f"the endpoint is claimed measured and no record is cited: {m}")
        cited = next(w for w in m.replace(",", " ").split()
                     if w.endswith("c1_decision.json"))
        record = REPO / cited.lstrip("-").strip()
        assert record.is_file(), f"the cited decision record is absent: {record}"
        verdict = json.loads(record.read_text())["verdict"]
        assert verdict in ("GO", "NO-GO", "INCONCLUSIVE"), verdict
        assert verdict in m, (
            f"the snapshot says {m!r} and the record says {verdict!r}")


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

    def test_the_stage_ladder_is_stated_exactly_once_and_in_full(self):
        """The other half of the stale-claim check above.

        Deleting a contradiction is not the same as stating the fact: a
        snapshot that had simply dropped `phase_c.c2.status` would satisfy
        every negative pattern while telling a reader nothing. The ladder is
        the one field a next-stage agent reads first, so it is checked
        structurally rather than by prose search.
        """
        assert snapshot()["stage_ladder"] == {
            "C0": "COMPLETE",
            "C1": "COMPLETE / GO",
            "C2": "CLOSED WITHOUT PROMOTION",
            "C3": "NOT STARTED",
            "next_scientific_stage": "C3",
        }

    @pytest.mark.parametrize("key,value,fact", [
        ("decision", "CLOSED WITHOUT PROMOTION", "the stage's disposition"),
        ("canonical_no_go_claimed", False, "that no canonical verdict is claimed"),
        ("new_incumbent_named_by_c2", None, "that C2 named no new incumbent"),
        ("probes_remaining", 0, "that no probes are owed"),
        ("further_scientific_spend", "NOT AUTHORIZED",
         "that no further C2 scientific spend is authorized"),
        ("observations_averaged_or_ranked", False,
         "that the two observations are neither averaged nor ranked"),
    ])
    def test_the_closure_states_each_fact_machine_readably(self, key, value, fact):
        """Parametrized so one missing field cannot hide behind five present
        ones -- and typed, because a prose field saying "no incumbent" is not
        something a consumer can branch on."""
        c = snapshot()["c2_closure"]
        assert key in c, f"the closure does not record {fact}"
        assert c[key] == value, f"{key} is {c[key]!r}, not {value!r}"

    def test_every_field_that_names_c2_s_position_agrees_with_the_ladder(self):
        """The contradiction that actually happened, checked across ALL owners.

        `stage_ladder`, `phase_c.c2`, `c2_closure`, `behavioural_session` and
        `latest_run` each state something about where C2 stands. Any one of
        them drifting is the defect; asking only the ladder would not have
        caught it, because the ladder was already correct while three other
        fields said C2 was running.
        """
        s = snapshot()
        closed = "CLOSED WITHOUT PROMOTION"
        for path, got in (
                ("stage_ladder.C2", s["stage_ladder"]["C2"]),
                ("phase_c.c2.status", s["phase_c"]["c2"]["status"]),
                ("c2_closure.decision", s["c2_closure"]["decision"]),
                ("behavioural_session.state", s["behavioural_session"]["state"]),
                ("latest_run._c2_state", s["latest_run"]["_c2_state"])):
            assert closed in got, f"{path} does not say {closed}: {got!r}"

        #: And the two independent "nothing is owed" owners agree numerically.
        assert (s["phase_c"]["c2"]["probes_owed"]
                == s["c2_closure"]["probes_remaining"] == 0)
        assert s["prepared_launch"]["any"] is False

        #: Structurally, not by prose. A mutation that re-added `phase_c.c2.owed`
        #: with different wording survived every phrase pattern above and the
        #: count check too -- the counts still said zero while a field named
        #: `owed` described the work. A key that names outstanding work is the
        #: claim, whatever it happens to say, so the closed stage may not carry
        #: one at all. `priced` goes with it: a price for work nobody owes.
        for gone in ("owed", "priced"):
            assert gone not in s["phase_c"]["c2"], (
                f"phase_c.c2 carries a `{gone}` field while the stage is "
                f"closed and owes nothing: {s['phase_c']['c2'][gone]!r}")
        #: C3 is not blocked on C2 naming an incumbent -- C2 named none, and
        #: the old precondition would read as C3 being permanently blocked.
        #: `c3` became a dict when the operator landed, matching `c1` and
        #: `c2`'s shape. Read the status field rather than the object: a
        #: `startswith` on the whole dict raises, and a `str(...)` around it
        #: would pass on any dict that happened to mention the phrase.
        c3 = s["phase_c"]["c3"]
        assert isinstance(c3, dict), "c3 is no longer a structured entry"
        assert c3["status"].startswith("NOT STARTED")
        #: The four facts §5 requires the machine-readable state to carry, so
        #: implementing the operator can never be mistaken for starting C3.
        assert "NO frozen seed set" in c3["status"]
        assert "ENGINEERING state only" in c3["operator"]
        assert "does NOT start formal C3" in c3["operator"]
        assert c3["batching_adoption_pilot"].startswith("AUTHORIZED")
        assert "NOT EXECUTED" in c3["batching_adoption_pilot"]
        assert "not formal C3" in c3["batching_adoption_pilot"]
        #: Over every field of the entry, not over `str(dict)`: the stale
        #: precondition could reappear in any one of them.
        assert not re.search(r"cannot start before C2 names",
                             " ".join(str(v) for v in c3.values()))

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
        #: C2 closed WITHOUT PROMOTION on 2026-09-24, and the snapshot kept
        #: saying it was running: `phase_c.c2.status` claimed "the behavioural
        #: selection among them is RUNNING: ten of twelve probes measured, two
        #: owed", `next` said "build and execute the attempt6 chain", and
        #: `behavioural_session` said "PROPOSAL ONLY" -- all while the
        #: top-level blocker already said the stage was closed and owed
        #: nothing. A reader had no way to tell which field was current.
        #:
        #: Each pattern targets the CLAIM, not the token: `attempt6` and
        #: `attempt15` still appear legitimately in fields that say the
        #: attempt6 grant confers nothing and that attempt15 must not happen,
        #: and a bare token search would refuse those negations too.
        (r"selection among them is RUNNING|C2[^.]{0,40}\bis RUNNING\b",
         "C2 is closed, not running"),
        (r"\btwo owed\b|still owes|owes (two|2) probes",
         "C2 owes no probes; all twelve were trained and scored"),
        (r"(build and )?execute the attempt6 chain|finish pre-staging",
         "no C2 launch is prepared and none may be"),
        (r"PROPOSAL ONLY", "the behavioural session ran and is closed"),
        (r"C2 is under execute-to-completion",
         "that authorization is spent and closed"),
    ])
    def test_the_stale_claim_is_gone(self, pattern, why):
        """The other half. A snapshot that states the new fact while still
        carrying the old one contradicts itself, and a reader has no way to
        tell which line is current."""
        bad = {p: v for p, v in strings(snapshot()) if re.search(pattern, v, re.I)}
        assert not bad, f"{why}; still present in:\n" + "\n".join(
            f"  {p}: {v}" for p, v in bad.items())

    def test_every_engineering_campaign_is_closed_and_priced(self):
        """There are TWO now, and a single `campaign_cost_usd` could only ever
        describe one of them.

        This read stage F's cost from a flat key. When the C2 full-search
        validation closed, the snapshot grew a second campaign and the flat key
        had to become per-validation — so the assertion is now over EVERY
        validation the snapshot names, and a new one that forgets to state its
        cost or its closure fails here rather than being silently uncounted.
        """
        cv = snapshot()["cuda_engineering_validation"]
        assert cv["authorizes"] == "nothing"
        named = {k: v for k, v in cv.items()
                 if not k.startswith("_") and k != "authorizes"}
        assert len(named) >= 2, (
            f"expected both CUDA validations, found {sorted(named)}")
        #: A TERMINAL verdict, not specifically a passing one. This demanded
        #: the literal "PASS", which reads as "state a verdict" only while
        #: every campaign happened to pass. The performance round did not: its
        #: three subruns each failed in the instrumentation rather than in the
        #: thing being measured, so it closed COMPLETE, and a guard that
        #: required the word PASS would have been an incentive to write it.
        TERMINAL = ("PASS", "COMPLETE", "FAIL", "TERMINATED", "ABANDONED")
        for key, text in named.items():
            assert any(v in text for v in TERMINAL), (
                f"{key} states no terminal verdict; one of {TERMINAL} is "
                "expected, and which one is a matter of what happened")
            assert "CLOSED" in text or "campaign CLOSED" in text, (
                f"{key} does not state that its campaign is closed")
            #: A dollar figure, so a campaign cannot be named without being
            #: priced -- the condition under which spend goes uncounted.
            assert re.search(r"\$\d+\.\d{4}", text), (
                f"{key} states no cost; an unpriced campaign is how "
                f"engineering spend stops reaching the project book")
            #: Every path it names must exist.
            for token in re.findall(r"logs/[\w./-]+", text):
                assert (REPO / token.rstrip(".,")).exists(), (
                    f"{key} names {token}, which is not in the tree")

    def test_an_approved_package_is_not_an_issued_authorization(self):
        """The distinction the whole launch contract rests on.

        This asserted `authorized.any is False` outright, which was the same
        sentence as "no package has been approved" only while none had. On
        2026-09-11 one was, and the invariant worth protecting is not that
        nothing is approved — it is that **an approval is not a launch**: a
        package permits a bounded number of one-use chains, and each chain still
        needs a grant, a launch-bound sweep, an issued authorization and a
        staged bundle before anything can be created.

        So the shape is pinned in whichever direction is true, and the usage is
        required to be counted and to stay inside a STATED bound — a package
        whose usage is untracked is a package without a limit.

        The bound was an attempt CAP, and this required one as an integer. The
        maintainer withdrew that cap on 2026-09-11: sessions are still counted,
        but what limits them is money. Requiring an integer cap would have
        forced the snapshot to state a limit that no longer exists, so the
        requirement is now that a bound is stated and checkable in whichever
        form is live — and when it is the money, the balance is required to be
        arithmetically consistent, which the count never was.
        """
        s = snapshot()
        a = s["authorized"]
        if not a["any"]:
            assert s["prepared_launch"]["any"] is False
            assert s["running"]["paid_compute"] is False
            return

        assert a.get("package_id"), (
            "the snapshot claims an authorization exists but names no package")
        used, cap = a.get("formal_sessions_used"), a.get("formal_sessions_max")
        assert isinstance(used, int), (
            "an approved package must count its sessions; uncounted usage "
            "cannot be checked against any limit")
        if isinstance(cap, int):
            assert 0 <= used <= cap, f"{used} of {cap} sessions used"
        else:
            assert cap is None, (
                f"formal_sessions_max is {cap!r}: an integer cap or an "
                "explicit null, never a missing key -- a cap that is absent "
                "because nobody wrote it reads exactly like one that was "
                "withdrawn")
            assert a.get("_sessions_max_meaning"), (
                "no cap and no explanation: a null limit must say that it is "
                "the amendment and name what binds instead")
            #: What binds instead, checked rather than described.
            b = s["budget"]
            spend, capped = b["cumulative_spend_usd"], b["authorized_cap_usd"]
            assert 0 < spend <= capped, f"{spend} of {capped}"
            assert b["remaining_usd"] == round(capped - spend, 4), (
                "the stated remainder is not the stated cap minus the stated "
                "spend")
            assert b.get("remaining_is_not_permission") is True
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

    def test_its_counts_are_the_same_numbers_the_snapshot_claims(self):
        """It hardcoded `10` and `9`, which had to be re-edited on every attempt
        and had already gone stale against the snapshot's own constants. The
        numbers now come from ONE place, so a count can drift in one document
        and be caught rather than duplicated correctly by hand."""
        live = self.live_region()
        C = TestTheAttemptCountIsOneNumber
        bad = [n for n in count_before(live, r"labels?") if n != C.LABELS]
        assert not bad, f"STATE.md claims {bad} labels, the count is {C.LABELS}"
        bad = [n for n in count_before(live, r"paid") if n != C.PAID]
        assert not bad, f"STATE.md claims {bad} paid, the count is {C.PAID}"

    def test_it_does_not_say_c1_was_never_measured(self):
        assert not re.search(r"NEVER MEASURED", self.live_region()), (
            "STATE.md's live region still says C1 was never measured; attempt "
            "9 passed both replay gates")

    def test_it_points_at_the_migration_record_rather_than_restating_it(self):
        assert "source-relocations/initialization-core/v1" in self.live_region()

    def test_the_current_view_carries_no_history_and_no_shelf(self):
        """One "current state" section, no superseded narrative, no archive.

        This test has had its second half REPLACED, and the reason matters. It
        used to also require that the superseded narrative still existed at
        `logs/archive/`, on the ground that deleting it would destroy the only
        record of what was believed then. The maintainer has since ruled that
        git history IS that record and that a superseded document kept in the
        working tree is one more thing that has to be stopped from reading as
        current -- so the archive is gone by decision, not by drift.

        What survives unchanged is the half that guards the live document: one
        section claims to be current, and no history is smuggled back into it.
        What replaces the other half is the rule that produced the change --
        nothing in `logs/` may become a shelf again.
        """
        text = STATE.read_text()
        heads = re.findall(r"^# Current state", text, re.M)
        assert len(heads) == 1, (
            f"{len(heads)} sections claim to be the current state")
        assert "# Superseded" not in text, (
            "STATE.md carries a superseded section again; history belongs to "
            "the experiment or stage that owns it, or to git")
        assert not (REPO / "logs/archive").exists(), (
            "logs/archive/ is back: a document that is merely out of date is "
            "deleted, and one that is still part of the record goes under its "
            "owner")

        #: The history that IS still part of the record is reachable, under the
        #: stage that owns it. Deleting that would be a real loss and this is
        #: what would notice.
        chronology = REPO / "logs/stages/stage-3/history/EXPERIMENTS.md"
        assert chronology.is_file(), (
            "the pre-layout experiment chronology is gone; it is the only "
            "record several Stage-3 experiments have")
        assert "stages/stage-3/history/EXPERIMENTS.md" in text, (
            "STATE.md does not lead a reader to the chronology")

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

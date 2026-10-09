"""Structural rules that keep documentation from rotting silently.

Every check here exists because the thing it forbids had actually happened. The
README carried a spend figure and a cap that were two raises out of date and an
"is not authorized" claim that had stopped being true; `REPO_LAYOUT.md` described
directories by name with nothing verifying they existed; `STATE.md` and
`current_state.json` had drifted into disagreeing about what was authorized.

None of these break a run. They break the next session's ability to trust what
it reads, which is worse, because it is discovered late and by inference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
STATE = REPO / "logs/state/current.md"
SNAPSHOT = REPO / "logs/state/current.json"
CATALOG = REPO / "logs/state/ownership.md"
LAYOUT = REPO / "docs/maintenance/REPO_LAYOUT.md"
POD_SCRIPTS = REPO / "docs/shared/POD_SCRIPTS.md"


def backticked(path: Path) -> set[str]:
    return set(re.findall(r"`([A-Za-z0-9_./*{}-]+)`", path.read_text()))


# --- the README owns no live facts ------------------------------------------

def test_the_readme_carries_no_live_spend_or_authorization_state():
    """It carried `$191.5462 against a $213.00 cap` through two cap raises, and
    "no `PhaseAAuthorization` artifact exists" after four had been issued.

    A README is read by people who will not check its date. Live facts belong to
    `current_state.json`, which is regenerated, and to `BUDGET_LEDGER.md`, which
    is append-only.
    """
    text = README.read_text()
    # Dollar amounts that look like a running total or a cap.
    money = [m for m in re.findall(r"\$[0-9]+\.[0-9]{2,4}", text)]
    assert not money, (
        f"the README states dollar amounts {money}; spend and caps are owned by "
        "logs/budget/ledger.md")

    forbidden = [
        # Naming the concept and linking to its owner is the desired shape;
        # what is forbidden is stating a VALUE here.
        r"cumulative spend[^.\n]*[0-9]",
        r"is (?:still )?(?:not )?authorized",
        r"no `?PhaseAAuthorization`? artifact",
        r"^### Current state",
    ]
    for pattern in forbidden:
        hit = re.search(pattern, text, re.M)
        assert not hit, (
            f"the README claims live state ({hit.group(0)!r}); that fact is "
            "owned by logs/state/current.json")


def test_the_readme_keeps_the_required_public_structure():
    """AGENTS.md §2.7 fixes the section list. Empty sections are allowed;
    missing ones are not."""
    text = README.read_text()
    for heading in ("Performance Trend and Project Goal", "How it works",
                    "Quick start", "Running the agent", "Project structure",
                    "Optim record history", "References", "Citation"):
        assert re.search(rf"^## .*{re.escape(heading)}", text, re.M), heading


def test_the_readme_points_at_the_owners_of_the_facts_it_dropped():
    text = README.read_text()
    for owner in ("logs/state/current.json", "logs/state/current.md",
                  "logs/budget/ledger.md", "logs/state/ownership.md",
                  "docs/maintenance/REPO_LAYOUT.md"):
        assert owner in text, f"the README does not point at {owner}"


# --- the layout describes a repository that exists --------------------------

def external_storage_roots() -> set[str]:
    """Host-local storage roots REPO_LAYOUT.md *declares* are outside the tree.

    Read from the document's own "Storage that is not in the tree" section
    rather than inferred from a leading slash, so an absolute path that wandered
    into some other section is still a defect rather than a free pass.
    """
    text = LAYOUT.read_text()
    start = text.index("## Storage that is not in the tree")
    end = text.index("\n## ", start)
    return {r for r in re.findall(r"`(/[A-Za-z0-9_./-]+)`", text[start:end])}


def test_every_path_named_in_the_repo_layout_exists():
    """A layout document nobody checks becomes a description of a repository
    that used to exist.

    **Repository-relative references must exist, always.** That is the whole
    point and it is not weakened below.

    Absolute references are a different kind of claim. The two the document
    names — the out-of-tree artifact store and the scratch area — are real
    operational facts about the maintainer host, and `docs/maintenance/REPO_LAYOUT.md`
    should keep naming them exactly. But `Path(REPO) / "/home/ecs-user/..."`
    **discards the base**, so the original assertion read the literal host
    filesystem and demanded that every execution environment be the maintainer's
    machine. A pod is not, and Phase-A attempt 8 paid $0.19 to discover it at the
    setup test gate after the staging, the frozen-asset gate and the RoPE check
    had all passed.

    So this test asks only PORTABLE questions, which every machine can answer the
    same way: every absolute path must be a DECLARED host-local storage root — a
    typo'd or invented one still fails — and every repository-relative reference
    must exist.

    Whether those declared roots exist HERE is not asked, because it is not a
    portable question. It used to be, via a trailing skip, and C1 attempt 6 was
    refused at the pod CPU gate partly for it: the test ran on the dev box and
    skipped on the pod, so the skip sets differed. Host-side existence is owned
    by `test_storage_inventory.py::test_the_registry_covers_the_out_of_tree_store`,
    which is host-scoped on purpose.
    """
    refs = {ref for ref in backticked(LAYOUT)
            if ("/" in ref or ref.endswith(".md"))
            and "*" not in ref and "{" not in ref}
    declared_external = external_storage_roots()

    absolute = {r for r in refs if r.startswith("/")}
    undeclared = sorted(absolute - declared_external)
    assert not undeclared, (
        f"REPO_LAYOUT.md names the absolute paths {undeclared} outside its "
        '"Storage that is not in the tree" section. An absolute path is a claim '
        "about a particular host; declare it as a storage area or make it "
        "repository-relative.")

    missing = sorted(r for r in refs - absolute if not (REPO / r).exists())
    assert not missing, f"REPO_LAYOUT.md names paths that do not exist: {missing}"

    # Deliberately NOT checked here: whether those declared host-local roots
    # EXIST on this machine. A paid pod is not required to contain
    # /home/ecs-user/aad-artifacts, and their legitimate absence must not turn a
    # PORTABLE contract test from PASS into SKIP.
    #
    # C1 attempt 6 was refused at the pod CPU gate partly for this: the test ran
    # on the dev box and skipped on the pod, an unexpected pod-only skip. The
    # paths are absolute literals in Markdown, so no fresh $HOME can reach them —
    # the fix is to stop asking a host question inside a portable test, not to
    # waive the difference.
    #
    # Host-side existence and inventory remain owned by
    # `test_storage_inventory.py::test_the_registry_covers_the_out_of_tree_store`,
    # which is correctly host-scoped and skips where the store is absent.


# --- STATE.md and current_state.json are one fact, two views ----------------

def load_snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def test_the_two_state_views_agree_on_money():
    """They drifted. The prose is a view of the JSON, so the numbers in it must
    be the JSON's numbers."""
    snap = load_snapshot()
    text = STATE.read_text()
    for key, fmt in (("cumulative_spend_usd", "{:.4f}"),
                     ("authorized_cap_usd", "{:.2f}"),
                     ("remaining_usd", "{:.4f}")):
        value = fmt.format(snap["budget"][key])
        assert value in text, (
            f"STATE.md does not show budget.{key} = {value}; the two views have "
            "drifted apart")


def test_the_two_state_views_agree_on_what_is_running_and_authorized():
    """Agreement, in whichever direction is true.

    This used to assert `paid_compute is False` outright — written when nothing
    had ever been launched, so "the two views agree" and "nothing is running"
    were the same sentence. They are not: once a pod is billing, that form
    demands the snapshot claim nothing is running while a pod costs money, which
    is the one failure mode a handoff document must never have.

    So it compares the two views instead of pinning one state. Quiet must be
    stated as quiet, and a live session must be stated as live — with the pod id,
    so the reader can go and look.
    """
    snap = load_snapshot()
    # Whitespace-normalised: the claims are prose and wrap wherever the sentence
    # happens to reach the margin. Matching raw text meant every reflow was a
    # test failure, and the two accepted newline positions were whichever two
    # had been written so far.
    #
    # And scoped to the CURRENT view. STATE.md keeps its superseded sections in
    # place, under headings that say so, and this test read the whole file — so
    # on 2026-09-11 "nothing is billing" and "nothing is prepared for launch"
    # were found only in the attempt-8 section frozen on 2026-09-06, and the
    # guard had been satisfied by history rather than by a claim about now. A
    # test that cannot tell a current statement from a recorded one is not
    # checking agreement between the two views; it is checking that the file
    # once contained a sentence.
    #: The whole file IS the current view: history lives with the experiment or
    #: stage that owns it, or in git. The split is checked by
    #: `test_the_current_view_carries_no_history_and_no_shelf`; what this needs
    #: is a current view that contains no history, so a claim found here cannot
    #: have been satisfied by a frozen section.
    current = STATE.read_text().split("\n# Superseded")[0]
    assert "# Superseded" not in STATE.read_text(), (
        "STATE.md carries history again; a claim about now could be satisfied "
        "by a frozen section instead of by a statement about now")
    text = " ".join(current.lower().split())

    if not snap["running"]["paid_compute"]:
        assert snap["running"]["pods"] == 0
        for claim in ("nothing is running", "nothing is billing"):
            assert claim in text, claim
    else:
        # A live run must be visible in prose, and identified.
        assert snap["running"]["pods"] >= 1
        assert snap["running"]["pod_id"].lower() in text, (
            "STATE.md does not name the pod the snapshot says is billing")
        for claim in ("nothing is running", "nothing is billing"):
            assert claim not in text, (
                f"STATE.md still says {claim!r} while a pod is billing")

    if not snap["authorized"]["any"]:
        assert "nothing is authorized" in text
    else:
        assert "nothing is authorized" not in text
    if not snap["prepared_launch"]["any"]:
        assert "nothing is prepared for launch" in text


def test_the_snapshot_stays_minimal_and_declares_its_contract():
    """It had grown to 33 KB and 28 keys by absorbing per-attempt history that
    already lived in the per-run directories.

    The ceiling moved 12_000 -> 13_000 and then -> 13_500, both on
    2026-09-27, and the reasons are recorded here rather than left as bare
    numbers. The snapshot began carrying a SECOND live authorization (the C3
    batching-adoption pilot) and then a THIRD (the packing-optimization
    pilot, stopped mid-chain with its screen measured), alongside the C1
    execution package and the operator's engineering state. All of that is
    current state, which this file owns, and none of it is history, which it
    does not.

    TWO RAISES IN ONE DAY IS ITSELF A SIGNAL. Before the next one, reclaim:
    a pilot that has closed with a verdict owns its figures in its own
    record and needs only its boundary here, which is what shrank the
    batching entry to one line. A raise for history would be the failure
    this guard exists to catch, so the shape check below asks that question
    directly instead of leaving the byte count to imply it.

    13_500 -> 14_000 on 2026-10-01, and the reclamation came FIRST, which is
    what this docstring asks for. Reclaimed in the same round: `phase_c.c3`
    went from attempt66's lost-checkpoint narrative to a closed phase's
    boundary once C3 returned its verdict (~1.1 KB), `phase_c2_replay` to
    three keys, `prepared_launch` to a note that no chain may be built, and
    three of the four CUDA validations to verdict + CLOSED + cost + owner.
    Two squeezes went TOO far and this file's own gates caught both: the
    attempt-4 consistency check requires the snapshot to record C2's Top-5
    ruling as ACCEPTED, and `test_the_fact_appears` requires stage F to keep
    `CONFIRMED ON REAL CUDA` and SHA `7027a8f4`. Both were restored.

    What that growth bought is THREE new live subjects, none of them history:
    the A-bsz3 adoption study, the scientific-vs-materialization identity
    precondition that decides whether a differing artifact may enter a later
    search, and the D-series directive. Each is a decision a reader acts on now.

    14_000 -> 15_500 on 2026-10-03, and the reclamation came first again. A
    FOURTH live subject arrived: D1 is designed, priced and blocked on two
    things a maintainer has to rule on, and a reader who does not see both
    blockers in the snapshot will not find them. Reclaimed in the same round:
    `a_bsz3` lost its restated verdict, replicate structure and pod count once
    A3 went terminal and its closeout became the owner (~0.1 KB), and the
    D-series entry carries pointers rather than the protocol — the design
    record owns every figure of it.

    THE RECLAMATION DID NOT COVER THE GROWTH, AND THAT IS THE HONEST READING.
    The file stood at 13_739 of 14_000 before this round, so 261 bytes of
    headroom had to absorb a fourth subject; it could not. The alternative was
    deleting live facts, which is precisely what the two squeezes recorded
    above got wrong. Before the next raise, ask whether Phase A's and Phase B's
    result blocks are still decisions anyone acts on, or whether the lineage
    they carry would read better as one line naming the incumbent.

    THAT QUESTION WAS THEN ASKED AND ANSWERED, AND THE ANSWER WAS HALF
    WRONG. Folding `phase_a_result` and `phase_b_result` into `phases` was
    justified as "nothing read the two removed keys", and for `phase_b_result`
    that WAS WRONG: eight assertions in
    `continuation_b/tests/test_continuation_b_one_probe_contract.py` read its
    status, winner, `winner_is_control`, `tie_break_ran`, `clears_by`,
    `read_with_care`, `not_capability` and `authorizes`. `phase_b_result` is
    restored, and `phases.phase_b` is a POINTER to it rather than a
    restatement, which is the shape to have chosen first: one fact, one owner.
    `phase_a_result` stays folded; grep finds no reader and that grep was run
    properly.

    THE SAME CLAIM WAS THEN MADE A SECOND TIME, AT THIS RECONCILIATION, BY AN
    AGENT WHO HAD ALREADY READ THE CORRECTION ABOVE. Two greps were run. The
    first matched `snapshot()[...]` and `snap[...]`; the readers use a local
    `state[...]`, so it found nothing. The second had the right pattern and
    would have found all eight — and ended in `| head`, which cut the output
    above the hits. **A grep for readers must not be piped into `head`, and
    must not assume the subscript spelling.** The restoration cost nothing this
    time only because the correction above was in the conflict being resolved.

    AND THE CORE SUITE CANNOT CATCH THIS ANY MORE. The eight readers are a
    continuation_b EXPERIMENT test, so after the test-boundary refactor they do
    not run in the default suite: a snapshot key can now lose its only reader's
    protection by being read outside `tests/`. That is what
    `test_every_snapshot_key_a_test_reads_still_exists` below is for.

    **The lesson is about ORDER, not about these two keys.** A reclamation is
    a code change to a file that gates read by name, so it owes the same
    verification as any other — before the round's last commit, not after it.

    The next place to look is `behavioural_session` and `phase_c2_replay`,
    whose subject C2 is CLOSED WITHOUT PROMOTION and whose figures `c2_closure`
    already owns — but check first which of them carries the word `ACCEPTED`,
    because the attempt-4 gate reads it and this file has already broken that
    gate three times.

    AND THE SAME SQUEEZE BROKE THE SAME GATE A THIRD TIME. Trimming
    `next_starting_point.status` dropped "with its Top-5 ACCEPTED and FROZEN",
    and `test_the_snapshot_does_not_contradict_itself_about_attempt_4` refused
    — exactly as the 2026-10-01 paragraph above records it refusing. Restored.
    A squeeze of this file is not a formatting change: two gates read specific
    words out of it, and they are the attempt-4 ACCEPTED ruling and stage F's
    `CONFIRMED ON REAL CUDA` with SHA `7027a8f4`. Check both by name before
    trimming, not after.

    TWO BRANCHES RAISED THIS CEILING INDEPENDENTLY FROM ONE BASE, AND THE
    RECONCILIATION ADDED NOTHING. `refactor/test-suite-boundary` went
    15_500 -> 16_000 for the test-suite boundary; `review/d1-target-aware` went
    15_500 -> 16_500 for D1's third blocker and the battery family. Neither
    knew about the other. The ceiling here is **16_500** — the higher of two
    decisions already made, not a fourth raise — and the union of both branches'
    subjects fits under it at 16_299 because the reclamation came first all
    three times. 201 bytes of headroom is thin on purpose: the next subject
    reclaims.

    Reclaimed on the two branches: `behavioural_session`, a second copy of
    `c2_closure`'s subject under the same owner document, with its two live
    facts (the 12-probe protocol, the 14 chains) moved into `c2_closure`; and
    `d_series.order`, because a maintainer order is history and `decisions.md`
    owns every order. `phase_c2_replay` is the block carrying `ACCEPTED` and
    was left alone, exactly as the warning above says.

    Reclaimed at the merge, 709 bytes, and it found a CONTRADICTION rather than
    just bytes: FIVE fields described why D1 is blocked, and two of them still
    said TWO blockers after a third appeared. `d_series.blockers` now owns the
    reasons; `blocker`, `next_starting_point.then` and `stage_ladder.D1` name
    the count and point there. D1's figures went to `d1_design.json` and
    `selection_noise.py`, which own them. `main_branch` and `test_suites`
    were two blocks on one subject and are now one.

    What the 17_000 buys is THREE live subjects, none of them history: the test
    suite has a boundary and a guard that holds it; D1's third blocker is a
    per-session envelope that binds SEPARATELY from the cumulative cap, so a
    grant moving only the cap still could not authorize the search; and the
    D-series battery FAMILY carries a new behavioural-distribution identity
    that must not be confused with `c1_confirmation`.

    `phase_c` is the largest block at ~2.6 KB and it is NOT the next place to
    look: a dozen gates read `c0.status`, `c1.measured`, `c2.status` and
    `c3.status` by name, so trimming it is the squeeze-breaks-a-gate failure
    recorded three times above, at the one block where it would break several
    at once. The next place to look is `d_series` once D1 is authorized or
    abandoned, because then its blockers stop being decisions anyone acts on.

    16_500 -> 18_000 on 2026-10-08, and the reclamation came first and did NOT
    cover the growth — which is the honest reading and is why the raise is
    recorded rather than the facts trimmed.

    What it buys is three live subjects in `d_series`, each one a decision a
    reader acts on now, and the first of them is why the other two exist:

    * **which checkpoint B is.** The design named C1's BEATEN arm (`c313d1b4`,
      `attention.weight_proxy_v0`) as its control for the whole of its design
      and search. C1 returned GO, so B is the TREATMENT at `53e30566` — and
      C1's delta between the two arms EXCEEDS the SESOI the D1 decision rule
      tests against, so the error could manufacture a GO rather than merely add
      noise. A reader who does not find this in the snapshot repeats it.
    * **what the behavioural rungs cost.** Arm materialization was one rebuild
      inside a FIXED session overhead — C3's figure, for a session with one
      arm — against D1 screening's five, so the priced session funded a fifth
      of the work. It is now a per-arm term.
    * **which seeds are bound.** They are `SHA256(design_hash + ...)` and the
      correction moved `design_hash`, so they were rebound prospectively. C0
      requires them bound before any candidate result exists.

    Reclaimed in the same round, each checked for readers BY NAME first — the
    paragraphs above record that claim being made wrongly twice, once because
    the grep was piped into `head`: `d_series.a3_precondition` (a closed
    precondition `a_bsz3.blocks` already owns), `d_series.scoring_identity`
    (folded into `scoring_protocol`, which three tests read and which owns the
    subject), and `d_series.family_and_sources` lost a construction commit the
    manifest owns. `prepared_launch.note` was also STALE — it spoke about C3
    while D1's search had since completed — and a stale note is worse than a
    long one.

    **The next place to look is still `d_series`, and the trigger is now
    explicit:** when D1 resumes and is authorized, its pricing and seeds move
    into the authorization that binds them and stop being snapshot facts.
    """
    snap = load_snapshot()
    assert snap["schema"] == "aadistill.current_state/v2"
    assert "_contract" in snap, "the snapshot does not say what it owns"
    assert len(SNAPSHOT.read_bytes()) < 18_000, (
        f"current_state.json is {len(SNAPSHOT.read_bytes())} bytes; it is the "
        "minimal snapshot, not an archive — history belongs in the per-run "
        "directories and decisions.md")
    #: THE THING THE BYTE COUNT IS A PROXY FOR. A snapshot absorbing history
    #: grows one key per attempt; a snapshot tracking current state does not.
    import re

    per_attempt = [k for k in snap
                   if re.search(r"attempt[_ ]?\d|run[_ ]?\d|_20\d{6}", k)]
    assert not per_attempt, (
        f"the snapshot has grown per-attempt keys {per_attempt}; those belong "
        "in the per-run directories, not in the live snapshot")
    #: And no entry may carry a per-subrun cost table: the campaign records
    #: own those, and copying one here is how 33 KB happened.
    import json as _json

    text = _json.dumps(snap)
    for marker in ("cost_usd", "elapsed_minutes", "subruns"):
        assert marker not in text, (
            f"the snapshot carries {marker!r}; per-subrun accounting belongs "
            "to the campaign records, not to the live snapshot")
    for key in ("budget", "frozen", "running", "authorized", "prepared_launch",
                "next_starting_point"):
        assert key in snap, key


def test_every_snapshot_key_a_test_reads_still_exists():
    """A reclaimed snapshot key must not have had a reader.

    Twice now a top-level key was folded away as "nothing reads it" when
    something did: eight assertions in a continuation_b test read
    `phase_b_result`. The first time cost a suite failure; the second time was
    caught only because the correction happened to be in the conflict being
    resolved.

    **Why a grep is not enough, and why this lives in CORE.** The readers spell
    it `state["phase_b_result"]` through a local variable, so a grep for
    `snapshot()[...]` misses them — and they are an EXPERIMENT test, so since
    the test-boundary refactor they do not run in the default suite at all. A
    key can now lose its only protection by being read outside `tests/`. This
    check is in core and scans the whole tree, which is the only place the
    question can be answered.

    Deliberately narrow: it reads subscripts of a few conventional snapshot
    variable names and asserts the key exists. It does not try to prove a
    reader's assertion still holds, and it does not forbid reclaiming a key —
    it forbids reclaiming one while a reader is left behind.
    """
    snap = load_snapshot()
    #: `state`, `snap`, `snapshot()` and `load_snapshot()` are the spellings the
    #: tree actually uses; a new one shows up as an unprotected key, not a pass.
    subscript = re.compile(
        r'(?:snapshot\(\)|load_snapshot\(\)|\bsnap\b|\bstate\b|\bcurrent\b)'
        r'\[\s*["\']([a-z][a-z0-9_]*)["\']\s*\]')
    orphaned: dict[str, list[str]] = {}
    here = Path(__file__).resolve()
    for root in ((REPO / "tests"), (REPO / "scripts" / "experiments")):
        for path in root.rglob("test_*.py"):
            #: Not this file: its own docstring quotes `state["phase_b_result"]`
            #: as the example, and a guard must not report its own prose.
            if path.resolve() == here:
                continue
            text = path.read_text()
            #: only files that actually open the snapshot
            if "current.json" not in text and "current_state" not in text:
                continue
            for key in set(subscript.findall(text)):
                if key not in snap and key in _KEYS_THE_SNAPSHOT_HAS_OWNED:
                    orphaned.setdefault(key, []).append(
                        str(path.relative_to(REPO)))
    assert not orphaned, (
        f"these tests read snapshot keys that no longer exist: {orphaned}. "
        "Reclaiming a key means moving its readers too, or the key stays.")


#: Top-level keys `current.json` has owned. A reader citing one of these is
#: asking the snapshot for it, so the key going missing is a broken contract
#: rather than an unrelated local variable. Add a name here when it is reclaimed.
_KEYS_THE_SNAPSHOT_HAS_OWNED = frozenset({
    "phase_a_result", "phase_b_result", "behavioural_session", "main_branch",
    "d_series", "phase_c", "c2_closure", "budget", "frozen", "stage_ladder",
    "blocker", "next_starting_point", "test_suites", "phases", "storage",
    "authorized", "running", "prepared_launch", "latest_run",
    "latest_verification", "architecture_migration", "phase_c2_replay",
    "cuda_engineering_validation", "a_bsz3",
})


def test_the_snapshot_carries_the_frozen_identities_unchanged():
    """The reorganization must not have edited a frozen hash into a new shape."""
    f = load_snapshot()["frozen"]
    assert f["science_plan_hash"] == (
        "02be33b9a7a8e26bc8bfb75795351e8cdc9ffd441b47066cc81887cfc511b55c")
    assert f["session_plan_hash"] == (
        "9377a2dc61f21790dd111d72a5de0e039ea1d31afef2d09e18c98a0b0cc2a0aa")
    assert f["stage3_evaluation_protocol_hash"] == (
        "250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4")
    assert f["equivalence_interval"] == pytest.approx(0.011695296982299022)
    assert f["feasibility_floor"] == pytest.approx(0.30)
    assert f["seeds"] == {"sa": 20260726, "sb": 20260801,
                          "sc_conditional": 20260813, "fourth_seed": "never"}


# --- everything is classified ----------------------------------------------

def test_every_log_is_classified_in_the_catalog():
    named = backticked(CATALOG)
    unclassified = []
    for path in sorted((REPO / "logs").iterdir()):
        name = path.name
        if name in named or f"{name}/" in named:
            continue
        # Families the catalog covers by statement rather than by name.
        if re.match(r"^e[0-9]", name) or name.endswith("_session_evidence.json"):
            continue
        unclassified.append(name)
    assert not unclassified, (
        f"logs/ entries with no class in CATALOG.md: {unclassified}. Every log "
        "is CURRENT, REFERENCE, HISTORICAL, SUPERSEDED or TERMINATED.")


def test_every_pod_script_is_classified():
    """The SHARED pod infrastructure is the catalogued set.

    Per-experiment drivers and launchers moved to their owning experiment
    directories in the 2026-10-08 information-architecture migration, and
    their classification is ownership itself — the directory says whose they
    are. What still needs a catalogue is the stage-neutral set every session
    shares.
    """
    named = backticked(POD_SCRIPTS)
    unclassified = [p.name for p in sorted((REPO / "scripts/shared/pod").iterdir())
                    if p.name not in named and p.name not in ("__pycache__",
                                                              "__init__.py")]
    assert not unclassified, (
        f"scripts/shared/pod entries with no class in POD_SCRIPTS.md: {unclassified}")


def test_the_device_canary_is_recorded_as_terminated_and_not_prepared():
    """It is kept for evidence and for its generic lesson. What must not happen
    is a future session finding the script and reading it as a plan."""
    text = POD_SCRIPTS.read_text()
    assert "TERMINATED — the paid device canary" in text
    assert "No further canary is prepared or authorized" in text
    snap = load_snapshot()
    assert any("canary" in a.lower() and "terminated" in a.lower()
               for a in snap["abandoned"]), (
        "current_state.json does not record the canary path as terminated")
    # And the evidence it was terminated *with* is still here.
    for d in ("logs/shared/validations/device-canary/runs/autoinit_device_canary_attempt1",
              "logs/shared/validations/device-canary/runs/autoinit_device_canary_attempt2"):
        assert (REPO / d).is_dir(), f"{d} was removed; that is paid evidence"


def test_the_obsolete_handoff_is_archived_and_bannered():
    archived = REPO / "docs/stages/stage-1/archive/HANDOFF_AUTOINITIALIZER_20260812.md"
    assert archived.is_file()
    assert not (REPO / "docs/HANDOFF_AUTOINITIALIZER.md").exists(), (
        "the superseded handoff is still in the live docs directory")
    head = archived.read_text()[:400]
    assert "ARCHIVED" in head and "Do not act on this document" in head


def _logs_tree_is_partial() -> bool:
    """Tracked `logs/` files absent from disk: a staged, pod-like checkout.

    OBSERVED rather than flagged. A pod receives a subset of the repository and
    the simulator models that by moving the rest aside, so a link from a present
    document to an absent one is a fact about the staging, not a broken
    reference. Keyed on a simulator variable this would be inverted on the pod,
    which does not set it.
    """
    import subprocess as _sp
    out = _sp.run(["git", "ls-files", "logs"], cwd=REPO, capture_output=True,
                  text=True, check=True).stdout.split()
    return any(not (REPO / f).exists() for f in out)


@pytest.mark.skipif(_logs_tree_is_partial(),
                    reason="logs/ is partially staged; an absent target is the "
                           "staging, not a broken link")
def _preserved(rel: str) -> bool:
    """Documents kept verbatim: a link in one records where a file WAS.

    The runs the index registers carry directory digests, so repointing a link
    inside one would change registered evidence to suit a relocation. The
    forward mapping lives in `logs/index.json`.`historical_paths` instead.
    """
    if _is_sealed_document(rel):
        return True
    return any(rel.startswith(d + "/") for d in _registered_run_dirs())


def _is_sealed_document(rel: str) -> bool:
    """A preregistration or proposal under `plans/`: a commitment whose value
    is that its bytes have not moved. Same convention as
    `scripts/maintenance/consolidation/fix_doc_links.is_sealed_document`,
    derived rather than imported so the core suite stays free of scripts/."""
    name = rel.rsplit("/", 1)[-1].lower()
    return ("/plans/" in rel and name.endswith(".md")
            and ("preregistration" in name or "proposal" in name))


def _registered_run_dirs() -> tuple[str, ...]:
    import json as _json
    idx = _json.loads((REPO / "logs/index.json").read_text())
    return tuple(rel for e in idx.get("runs", [])
                 for rel in (e.get("components") or {}).values()
                 if (REPO / rel).is_dir())


def test_no_markdown_link_points_at_a_file_that_is_not_there():
    """Cross-references are what replaces a duplicated copy, so a broken one is
    a lost fact rather than a cosmetic defect. Two whole classes of these existed
    until 2026-08-18: docs/stages/stage-1/AUTOINIT_REFERENCE.md linked to `logs/` files as if
    they were siblings, and the handoff kept `../logs/…` after being moved a
    directory deeper."""
    broken = []
    for md in sorted(REPO.rglob("*.md")):
        if any(part in (".git", ".venv", "__pycache__") for part in md.parts):
            continue
        if _preserved(md.relative_to(REPO).as_posix()):
            continue
        for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", md.read_text(errors="ignore")):
            target = m.group(1).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / target).exists():
                broken.append(f"{md.relative_to(REPO)} -> {target}")
    assert not broken, f"markdown links pointing nowhere: {broken}"

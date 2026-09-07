"""Whether one more full attempt fits, DERIVED — never hand-written.

On 2026-09-07 the attempt-9 closeout wrote, in three documents at once:

    `$15.9002` uncommitted — which is **below the `$15.1475` per-attempt ceiling
    by only `$0.7527`**, so a further full attempt no longer fits under the cap.

Both halves of that sentence are in it: `15.9002 - 15.1475 = +0.7527`, which is
the amount by which it CLEARS the ceiling, not the amount by which it falls
short. The conclusion is the negation of its own arithmetic. It was reviewed by
a human and by me and neither caught it, because "clears by only $0.75" and "no
longer fits" both sound like the same pessimism.

The consequence is not academic. A maintainer reading "a full attempt no longer
fits" would conclude Phase C1 is over for budget reasons, when in fact exactly
one ceiling-sized attempt still fits — leaving `$0.7527` and permitting no
second one.

So this module derives the two quantities from the three recorded inputs and
requires every prose claim to agree with the derivation. It deliberately does
not re-state the answer: a test that hard-codes `full_attempt_fits = True` would
have to be edited on the next spend, which is exactly when it stops being
checked. The inputs are the snapshot's cumulative spend and cap, and the
ceiling from `logs/phase_c1_pricing.json`.

Scope: money arithmetic and claims about it. Whether an attempt is AUTHORIZED is
a maintainer decision and is not derivable from any number here — headroom has
never been permission.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "logs/current_state.json"
PRICING = REPO / "logs/phase_c1_pricing.json"
LEDGER = REPO / "logs/BUDGET_LEDGER.md"
STATE = REPO / "logs/STATE.md"

#: Prose that asserts a full attempt does NOT fit. Each is a claim of
#: impossibility, so a document may carry one only when the arithmetic agrees.
DOES_NOT_FIT_CLAIMS = (
    "no longer fits",
    "does not fit",
    "no longer covers a full",
    "does not cover a full",
    "no full attempt fits",
    "cannot fund a full attempt",
    "below the per-attempt ceiling",
)

#: Two deliberate narrowings, each with a reason and each tested below.
#:
#: 1. Only text NEAR the ceiling figure counts. The ledger has an entry from
#:    2026-08 reading "16.9 GiB does not fit in 66% of 20.3 GiB free" — a true
#:    statement about DISK. A guard that fires on it teaches the next agent to
#:    delete accurate history to get a green suite.
#: 2. Text inside a marked CORRECTION is quotation, not assertion. A correction
#:    that may not quote the sentence it corrects is a correction that erases
#:    the record, which is the opposite of what P11 asks for.
#:
#: Both narrowings are holes if they are wider than stated, so
#: `test_the_narrowings_do_not_swallow_a_live_claim` mutates a live claim into
#: each of them and requires it to still be caught.
CEILING_WINDOW = 260


def _correction_free(text: str) -> str:
    """Drop marked corrections: they QUOTE the error they are correcting.

    Inline `*(Corrected …)*` spans go FIRST, and only then are blockquote blocks
    that still carry the marker dropped whole. The order matters: STATE.md keeps
    each attempt as one long blockquote, so doing it the other way round would
    let a single inline correction suppress the entire entry around it — a far
    wider exemption than the one being asked for. Nothing else is removed; an
    unmarked sentence is always a live claim.
    """
    text = re.sub(r"\*\(\s*corrected.*?\)\*", " ", text, flags=re.I | re.S)

    kept, block = [], []

    def flush():
        if block and not any("corrected" in ln.lower() for ln in block):
            kept.extend(block)
        block.clear()

    for line in text.splitlines():
        if line.lstrip().startswith(">"):
            block.append(line)
            continue
        flush()
        kept.append(line)
    flush()
    return "\n".join(kept)


def _windows_around_the_ceiling(path: Path, ceiling: float) -> list[str]:
    """Normalised text near every mention of the per-attempt ceiling."""
    text = " ".join(_correction_free(path.read_text()).lower().split())
    needle = f"{ceiling:.4f}"
    return [text[max(0, m.start() - CEILING_WINDOW):m.end() + CEILING_WINDOW]
            for m in re.finditer(re.escape(needle), text)]


def _inputs() -> tuple[float, float, float]:
    snap = json.loads(SNAPSHOT.read_text())["budget"]
    ceiling = json.loads(PRICING.read_text())["totals"]["hard_ceiling_usd"]
    return (float(snap["cumulative_spend_usd"]), float(snap["authorized_cap_usd"]),
            float(ceiling))


def _derive() -> dict:
    cumulative, cap, ceiling = _inputs()
    remaining = round(cap - cumulative, 4)
    reserve_after_ceiling = round(cap - cumulative - ceiling, 4)
    return {
        "cumulative": cumulative, "cap": cap, "ceiling": ceiling,
        "remaining": remaining,
        "reserve_after_ceiling": reserve_after_ceiling,
        #: The whole point. `<=`, because spending exactly to the cap is within
        #: it; a strict `<` would refuse an attempt that lands on the number.
        "full_attempt_fits": cumulative + ceiling <= cap,
    }


def test_the_snapshot_remaining_is_the_cap_minus_the_spend():
    d = _derive()
    snap = json.loads(SNAPSHOT.read_text())["budget"]
    assert round(float(snap["remaining_usd"]), 4) == d["remaining"], (
        f"remaining_usd {snap['remaining_usd']} != cap {d['cap']} - cumulative "
        f"{d['cumulative']} = {d['remaining']}")


def test_fitting_and_the_reserve_are_consistent_by_construction():
    """The two derived quantities cannot disagree with each other."""
    d = _derive()
    assert d["full_attempt_fits"] == (d["reserve_after_ceiling"] >= 0), (
        f"fits={d['full_attempt_fits']} but reserve={d['reserve_after_ceiling']}")
    assert d["reserve_after_ceiling"] == round(d["remaining"] - d["ceiling"], 4)


def test_no_document_claims_a_full_attempt_does_not_fit_when_it_does():
    """The regression. Prose may not contradict the arithmetic above it."""
    d = _derive()
    if not d["full_attempt_fits"]:
        return                       # the claim would be TRUE; nothing to check

    offenders = []
    for path in (SNAPSHOT, LEDGER, STATE):
        for window in _windows_around_the_ceiling(path, d["ceiling"]):
            for claim in DOES_NOT_FIT_CLAIMS:
                if claim in window:
                    offenders.append(
                        f"{path.relative_to(REPO)}: {claim!r} in ...{window}...")
    assert not offenders, (
        f"cumulative {d['cumulative']} + ceiling {d['ceiling']} = "
        f"{round(d['cumulative'] + d['ceiling'], 4)} <= cap {d['cap']}, so ONE "
        f"full attempt fits and leaves ${d['reserve_after_ceiling']}. These "
        "documents say otherwise:\n  " + "\n  ".join(offenders))


def test_no_boolean_field_denies_a_fit_the_arithmetic_allows():
    """The same rule for machine-readable claims, not only prose.

    A future snapshot may record the verdict as a field. If it does, it must be
    the derived one — a stored boolean is a copy, and copies go stale.
    """
    d = _derive()
    snap = json.loads(SNAPSHOT.read_text())["budget"]
    for key in ("full_attempt_fits", "a_full_attempt_fits", "fits_one_attempt"):
        if key in snap:
            assert bool(snap[key]) == d["full_attempt_fits"], (
                f"budget.{key} = {snap[key]!r}, derived {d['full_attempt_fits']}")
    for key in ("reserve_after_ceiling_usd", "reserve_usd"):
        if key in snap:
            assert round(float(snap[key]), 4) == d["reserve_after_ceiling"], (
                f"budget.{key} = {snap[key]!r}, derived "
                f"{d['reserve_after_ceiling']}")


def test_a_stated_reserve_is_never_presented_as_a_shortfall():
    """`$0.7527` is headroom. It was written as though it were a deficit.

    The specific mistake: a POSITIVE difference described with the language of a
    shortfall ("below … by only"). If a document quotes the reserve figure, it
    must not be inside a sentence that says the money is short.
    """
    d = _derive()
    if not d["full_attempt_fits"]:
        return
    reserve = f"{d['reserve_after_ceiling']:.4f}"
    offenders = []
    for path in (SNAPSHOT, LEDGER, STATE):
        text = " ".join(_correction_free(path.read_text()).lower().split())
        for m in re.finditer(re.escape(reserve), text):
            window = text[max(0, m.start() - 160):m.end() + 160]
            for phrase in ("below the", "short of", "falls short", "not enough"):
                if phrase in window:
                    offenders.append(
                        f"{path.relative_to(REPO)}: {phrase!r} near the "
                        f"${reserve} reserve: ...{window}...")
    assert not offenders, (
        f"${reserve} is what REMAINS after one full attempt, not a shortfall:\n  "
        + "\n  ".join(offenders))


def test_the_derivation_would_catch_the_mistake_it_was_written_for():
    """Mutation, in-process: the attempt-9 numbers with the wrong conclusion.

    A guard whose triggering case never runs is a guard nobody has seen work.
    """
    cumulative, cap, ceiling = 267.8598, 283.7600, 15.1475
    assert cumulative + ceiling <= cap, "the historical case must be a FIT"
    assert round(cap - cumulative - ceiling, 4) == 0.7527

    text = ("$15.9002 uncommitted — which is below the $15.1475 per-attempt "
            "ceiling by only $0.7527, so a further full attempt no longer fits "
            "under the cap.")
    hits = [c for c in DOES_NOT_FIT_CLAIMS if c in text]
    assert hits, "the claim list does not match the sentence it exists to catch"


def test_the_narrowings_do_not_swallow_a_live_claim(tmp_path):
    """Mutation: put the real sentence in each narrowing and require a catch.

    A scope reduction is only safe if the thing it excludes is genuinely not the
    thing being looked for. So the offending sentence is placed (a) far from any
    ceiling figure, (b) inside a marked correction, and (c) plainly, and the
    behaviour of each is asserted.
    """
    claim = ("$15.9002 uncommitted — below the $15.1475 per-attempt ceiling by "
             "only $0.7527, so a further full attempt no longer fits.")

    plain = tmp_path / "plain.md"
    plain.write_text(f"# ledger\n\n{claim}\n")
    hits = [w for w in _windows_around_the_ceiling(plain, 15.1475)
            if any(c in w for c in DOES_NOT_FIT_CLAIMS)]
    assert hits, "a plain live claim is not caught; the guard is inert"

    quoted = tmp_path / "quoted.md"
    quoted.write_text(f"# ledger\n\n> **CORRECTED 2026-09-08.** It read: {claim}\n")
    assert not [w for w in _windows_around_the_ceiling(quoted, 15.1475)
                if any(c in w for c in DOES_NOT_FIT_CLAIMS)], (
        "a marked correction is quotation and must not be read as a claim")

    #: The correction exclusion is per-BLOCK, so a live claim OUTSIDE the block
    #: survives even when a correction sits next to it.
    mixed = tmp_path / "mixed.md"
    mixed.write_text(f"# ledger\n\n> **CORRECTED.** It read: {claim}\n\n{claim}\n")
    assert [w for w in _windows_around_the_ceiling(mixed, 15.1475)
            if any(c in w for c in DOES_NOT_FIT_CLAIMS)], (
        "the correction block swallowed a live claim beside it")

    #: And the disk sentence the real ledger carries stays clear, because it is
    #: nowhere near a ceiling figure.
    disk = tmp_path / "disk.md"
    disk.write_text("16.9 GiB does not fit in 66% of 20.3 GiB free\n")
    assert not _windows_around_the_ceiling(disk, 15.1475)


def test_an_inline_correction_does_not_exempt_the_entry_around_it(tmp_path):
    """STATE.md's shape: one long blockquote with an inline correction in it.

    The inline span must be removed WITHOUT exempting its neighbours, or a whole
    attempt entry silently stops being checked.
    """
    claim = ("$15.9002 uncommitted — a further full attempt no longer fits "
             "against the $15.1475 ceiling.")
    doc = tmp_path / "state.md"
    doc.write_text(
        "> **Attempt 9.** Cumulative $267.8598 of $283.7600, leaving $15.9002.\n"
        "> *(Corrected 2026-09-08: this line previously said $15.9002 does not "
        "cover a full $15.1475 attempt.)*\n"
        f"> {claim}\n")
    hits = [w for w in _windows_around_the_ceiling(doc, 15.1475)
            if any(c in w for c in DOES_NOT_FIT_CLAIMS)]
    assert hits, (
        "an inline correction exempted the live claim beside it in the same "
        "blockquote")

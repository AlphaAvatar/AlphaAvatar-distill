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
STATE = REPO / "logs/STATE.md"
SNAPSHOT = REPO / "logs/current_state.json"
CATALOG = REPO / "logs/CATALOG.md"
LAYOUT = REPO / "docs/REPO_LAYOUT.md"
POD_SCRIPTS = REPO / "docs/POD_SCRIPTS.md"


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
        "logs/BUDGET_LEDGER.md")

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
            "owned by logs/current_state.json")


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
    for owner in ("logs/current_state.json", "logs/STATE.md",
                  "logs/BUDGET_LEDGER.md", "logs/CATALOG.md",
                  "docs/REPO_LAYOUT.md"):
        assert owner in text, f"the README does not point at {owner}"


# --- the layout describes a repository that exists --------------------------

def test_every_path_named_in_the_repo_layout_exists():
    """A layout document nobody checks becomes a description of a repository
    that used to exist."""
    missing = sorted(
        ref for ref in backticked(LAYOUT)
        if ("/" in ref or ref.endswith(".md"))
        and "*" not in ref and "{" not in ref
        and not (REPO / ref).exists())
    assert not missing, f"REPO_LAYOUT.md names paths that do not exist: {missing}"


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
    snap = load_snapshot()
    text = STATE.read_text().lower()
    assert snap["running"]["paid_compute"] is False
    assert snap["authorized"]["any"] is False
    assert snap["prepared_launch"]["any"] is False
    for claim in ("nothing is running", "nothing is billing",
                  "nothing is authorized", "nothing is\nprepared for launch"):
        assert claim in text or claim.replace("\n", " ") in text, claim


def test_the_snapshot_stays_minimal_and_declares_its_contract():
    """It had grown to 33 KB and 28 keys by absorbing per-attempt history that
    already lived in the per-run directories."""
    snap = load_snapshot()
    assert snap["schema"] == "aadistill.current_state/v2"
    assert "_contract" in snap, "the snapshot does not say what it owns"
    assert len(SNAPSHOT.read_bytes()) < 12_000, (
        f"current_state.json is {len(SNAPSHOT.read_bytes())} bytes; it is the "
        "minimal snapshot, not an archive — history belongs in the per-run "
        "directories and decisions.md")
    for key in ("budget", "frozen", "running", "authorized", "prepared_launch",
                "next_starting_point"):
        assert key in snap, key


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
    named = backticked(POD_SCRIPTS)
    unclassified = [p.name for p in sorted((REPO / "scripts/pod").iterdir())
                    if p.name not in named]
    assert not unclassified, (
        f"scripts/pod entries with no class in POD_SCRIPTS.md: {unclassified}")


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
    for d in ("logs/autoinit_device_canary_attempt1",
              "logs/autoinit_device_canary_attempt2"):
        assert (REPO / d).is_dir(), f"{d} was removed; that is paid evidence"


def test_the_obsolete_handoff_is_archived_and_bannered():
    archived = REPO / "docs/archive/HANDOFF_AUTOINITIALIZER_20260812.md"
    assert archived.is_file()
    assert not (REPO / "docs/HANDOFF_AUTOINITIALIZER.md").exists(), (
        "the superseded handoff is still in the live docs directory")
    head = archived.read_text()[:400]
    assert "ARCHIVED" in head and "Do not act on this document" in head

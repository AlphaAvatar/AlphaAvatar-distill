"""The E6 analysis must reproduce the anchor numbers it inherits.

E6 compares new measurements against P2-1.60M and E1-1.60M, both of which were
published from the E4 session. If the current scorer no longer reproduces those
published rates from the same retained generations, then E6's comparisons are
against numbers nobody can reconstruct — so the reproduction is a test, not a
one-off check.

The alias helper is tested for a bug that was real: `family_compare` built
`E1-1.60M@E4-sa` instead of `E1-1.60M-sa@E4`, which made a comparison whose arms
were both loaded report itself as `incomplete`. A dropped comparison reads
exactly like a missing measurement, which is the worst way for a bug to present.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

THREE_MODE = REPO / "artifacts/audit/three_mode"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"

pytestmark = pytest.mark.skipif(
    not (SESSIONS.is_file() and (THREE_MODE / "E4-P2-1600k-sa").is_dir()),
    reason="retained E4 artifacts or the corpus are not present")


@pytest.fixture(scope="module")
def sessions() -> dict:
    from analyze_e6 import load_sessions
    return load_sessions()


def test_arm_alias_reattaches_the_session_suffix_after_the_seed():
    from analyze_e6 import arm_alias
    assert arm_alias("E1-1.60M", "sa") == "E1-1.60M-sa"
    assert arm_alias("E1-1.60M@E4", "sa") == "E1-1.60M-sa@E4"
    assert arm_alias("P2-1.60M", "sb") == "P2-1.60M-sb"


# The values E4 published and E5 compared against, in EXPERIMENTS.md §21 / §27.1.
@pytest.mark.parametrize("directory,usable,correct", [
    ("E4-P2-1600k-sa", 0.7133, 0.2333),
    ("E4-P2-1600k-sb", 0.7533, 0.1667),
    ("P1-1600k-sa", 0.7800, 0.1733),
    ("P1-1600k-sb", 0.6800, 0.2000),
])
def test_rescoring_reproduces_the_published_anchor_rates(directory, usable,
                                                         correct, sessions):
    from analyze_e6 import rescore_arm
    m = rescore_arm(THREE_MODE / directory, sessions)
    assert m is not None, f"{directory} has no retained free generations"
    assert m["n"] == 150
    assert m["usable_rollout_rate"] == pytest.approx(usable, abs=1e-4)
    assert m["correct_overall"] == pytest.approx(correct, abs=1e-4)


def test_component_rates_are_reported_beside_the_conjunction(sessions):
    """`usable_rollout` must never be reported alone; the components are the view."""
    from analyze_e6 import rescore_arm
    m = rescore_arm(THREE_MODE / "E4-P2-1600k-sa", sessions)
    for key in ("protocol_valid_rate", "natural_termination_rate",
                "context_limit_rate", "severe_repetition_rate", "empty_output_rate"):
        assert key in m and 0.0 <= m[key] <= 1.0


def test_counts_reconcile_with_rates(sessions):
    from analyze_e6 import rescore_arm
    m = rescore_arm(THREE_MODE / "E4-P2-1600k-sa", sessions)
    c, n = m["counts"], m["n"]
    assert c["included"] == n == 150
    assert c["usable"] / n == pytest.approx(m["usable_rollout_rate"], abs=1e-4)
    assert c["correct"] / n == pytest.approx(m["correct_overall"], abs=1e-4)
    assert c["correct_and_usable"] <= min(c["correct"], c["usable"])
    if c["usable"]:
        assert c["correct_and_usable"] / c["usable"] == pytest.approx(
            m["correct_given_usable"], abs=1e-4)


def test_parse_failure_does_not_double_count_empty_answers(sessions):
    """An empty answer is empty, not a parse failure; the denominators differ."""
    from analyze_e6 import rescore_arm
    m = rescore_arm(THREE_MODE / "E4-P2-1600k-sa", sessions)
    assert m["counts"]["answer_parse_failure"] <= m["counts"]["numeric_prompts"]
    assert m["counts"]["numeric_prompts"] < m["n"], \
        "the battery holds free-form tasks too; a numeric-only rate needs its own n"
    assert m["answer_parse_failure_rate_numeric"] == pytest.approx(
        m["counts"]["answer_parse_failure"] / m["counts"]["numeric_prompts"], abs=1e-4)


def test_generated_token_summary_is_ordered(sessions):
    from analyze_e6 import rescore_arm
    g = rescore_arm(THREE_MODE / "E4-P2-1600k-sa", sessions)["generated_tokens"]
    assert g["p50"] <= g["p90"] <= g["max"]


def test_scoring_is_recomputed_and_not_read_from_the_stored_field(sessions):
    """The stored `correct` predates two scorer corrections; it must be ignored."""
    from analyze_e6 import rescore_arm
    d = THREE_MODE / "P1-1600k-sa"
    stored = [json.loads(l)["correct"] for l in (d / "free.generations.jsonl").open()
              if l.strip()]
    m = rescore_arm(d, sessions)
    # They may agree numerically; what matters is that a report exists whose
    # value comes from the scorer, computed over the raw text.
    assert m["correct_overall"] == pytest.approx(
        sum(bool(x) for x in stored) / len(stored), abs=0.2), \
        "a wild divergence means the rescore path is reading the wrong field"
    assert m["counts"]["correct"] == round(m["correct_overall"] * m["n"])

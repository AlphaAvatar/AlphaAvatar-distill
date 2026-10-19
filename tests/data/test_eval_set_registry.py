"""The repository's permanent evaluation sets are intact and their scoring is unchanged.

`data/eval-set/` is read by formal measurement in every stage (AGENTS.md P10.3).
This is the check that makes an unauthorized change to it LOUD: any edit to an
item, a gold answer, a manifest or a scorer's verdict on the registered
conformance vectors fails here, and so in CI. A legitimate change appends a new
identity to the registry citing a maintainer authorization -- which this test
also checks exists as a record.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from aadistill.evaluation import eval_registry as ER
from shared import eval_sets as ES

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "logs/budget/decisions.md"

pytestmark = pytest.mark.skipif(not ES.registry_path().is_file(),
                                reason="no evaluation registry in this checkout")


def _sets():
    return [e["eval_set_id"] for e in ES.registry()["eval_sets"]]


def test_default_formal_set_is_registered():
    assert ES.default_formal() in _sets()


@pytest.mark.parametrize("set_id", _sets() if ES.registry_path().is_file() else [])
def test_registered_set_matches_its_authorized_identity(set_id):
    assert ER.verify(ES.registry_path(), set_id) == []


@pytest.mark.parametrize("set_id", _sets() if ES.registry_path().is_file() else [])
def test_every_identity_cites_an_existing_decision_record(set_id):
    headings = set(re.findall(r"^## (\d{4}-\d{2}-\d{2}) — (.+)$", DECISIONS.read_text(), re.M))
    for h in ER.entry(ES.registry(), set_id)["identity_history"]:
        m = re.match(r"logs/budget/decisions\.md (\d{4}-\d{2}-\d{2}) \((.+)\)$", h["authorization"])
        assert m, f"authorization must cite a decisions.md heading: {h['authorization']!r}"
        assert (m.group(1), m.group(2)) in headings, f"no such decision record: {m.groups()}"


@pytest.mark.parametrize("set_id", _sets() if ES.registry_path().is_file() else [])
def test_scoring_semantics_reproduce_the_conformance_vectors(set_id):
    pytest.importorskip("sympy")   # the math scorer degrades to string comparison without it
    from shared.evaluation.eval_set_scoring import conformance
    result = conformance(set_id)
    assert result, "no conformance vectors registered"
    bad = {c: r for c, r in result.items() if r["differences"]}
    assert not bad, bad


def test_held_out_units_are_reserved_by_the_corpus_builder():
    held = ES.held_out_paths()
    assert held and all((REPO / p).is_file() for p in held)
    assert not any("legacy_150" in p for p in held)     # in-sample units are training data


# --- eval-set-1's frozen formal core (maintainer decision 2026-10-19) -----------

def _es1():
    return ES.manifest("eval-set-1")


def test_eval_set_1_is_frozen_with_its_formal_core():
    e = ER.entry(ES.registry(), "eval-set-1")
    assert e["status"] == "frozen" and len(e["identity_history"]) >= 2
    c = _es1()["components"]
    v = c["fluid_reasoning_720"]["validity"]
    assert (v["formal"], v["formal_static"], v["formal_interactive"], v["excluded"]) == (714, 570, 144, 6)
    core = c["stage3_capability_846"]["core"]
    assert (core["behaviour_items"], core["behaviour_core"], core["behaviour_core_held_out"]) == (76, 69, 65)
    for comp in c.values():
        assert comp["scorer"]["formal_protocol"] == "v2"


def test_the_report_scores_exactly_the_formal_rows(tmp_path):
    """Gold answers for every fluid item: the formal row counts 714, all correct,
    and the six excluded items are reported beside it, never inside it. The
    episodes replay the set's own registered v2 oracle vectors (data, so this
    core test reaches into no experiment's code)."""
    import json
    import shared.evaluation.eval_set_scoring as S

    root = ES.set_root("eval-set-1")
    oracle = {}
    for line in (root / "conformance/fluid_reasoning_720.v2.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["variant"] == "oracle_markdown_actions_generated_v2":
            oracle[r["id"]] = r["input"]
    rows = []
    for t in ES.items("eval-set-1", "fluid_reasoning_720", "dev_v1"):
        rows.append({"task_id": t["task_id"], **oracle[t["task_id"]]} if t["interactive"] else
                    {"task_id": t["task_id"], "end": "eos",
                     "turns": [S.synthetic_turn(f"\\boxed{{{t['answer']}}}")]})
    (tmp_path / "fluid_reasoning_720__dev_v1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    rep = S.report(tmp_path, "gold", {}, "eval-set-1", rows_out=tmp_path / "rows")
    unit = rep["units"]["fluid_reasoning_720/dev_v1"]
    assert (unit["n_items"], unit["correct"]) == (714, 714)
    assert unit["all_registered_items"]["n_items"] == 720
    assert sum(unit["excluded_from_formal"].values()) == 6
    flagged = [json.loads(l) for l in (tmp_path / "rows/fluid_reasoning_720__dev_v1.rows.jsonl")
               .read_text().splitlines()]
    assert len(flagged) == 720 and sum(not r["formal"] for r in flagged) == 6
    assert rep["units"]["stage3_capability_846/gsm8k"]["status"] == "not_run"


def test_the_report_reads_generations_whose_text_holds_unicode_line_separators(tmp_path):
    """`json.dumps(ensure_ascii=False)` writes U+2028/U+2029/U+0085 raw inside a
    generation, and `str.splitlines()` breaks a JSONL record at each of them:
    Stage-0 dose d1's report died on exactly that. Records are split on "\\n"."""
    import json
    import shared.evaluation.eval_set_scoring as S

    items = ES.items("eval-set-1", "stage3_capability_846", "gsm8k")
    with (tmp_path / "stage3_capability_846__gsm8k.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps({"id": it["id"], "think_preopened": False,
                                "natural_termination": True, "degeneration_triggered": False,
                                "degeneration_kind": None, "stop_reason": "eos",
                                "raw": "<think>\na b c\x85d\n</think>\n\n1<|im_end|>"},
                               ensure_ascii=False) + "\n")
    unit = S.report(tmp_path, "sep", {}, "eval-set-1")["units"]["stage3_capability_846/gsm8k"]
    assert (unit["status"], unit["n"], unit["missing"]) == ("measured", len(items), 0)

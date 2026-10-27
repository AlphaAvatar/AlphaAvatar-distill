"""Research A's records agree with the code, the tokenizer and the evidence they cite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
IDENTITY = REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json"
DESIGN = REPO / "configs/stages/stage-2/teacher_modes/design.json"
PREREG = REPO / "logs/stages/stage-2/teacher_modes/plans/preregistration.json"
AUDIT = REPO / "logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json"
FORECAST = REPO / "logs/stages/stage-2/teacher_modes/plans/cost_forecast.json"


def load(p):
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def hybrid():
    from transformers import AutoTokenizer

    d = load(IDENTITY)
    try:
        return AutoTokenizer.from_pretrained(d["model"], revision=d["revision"], local_files_only=True)
    except Exception:
        pytest.skip("hybrid tokenizer not in the local HF cache")


def test_the_identity_record_is_the_tokenizer_it_names(hybrid):
    from aadistill.data.token_documents import generation_suffix
    from aadistill.models.teacher import tokenizer_hash

    d = load(IDENTITY)
    assert hashlib.sha256(hybrid.chat_template.encode()).hexdigest() == d["chat_template_sha256"]
    assert tokenizer_hash(hybrid) == d["vocabulary_sha256"]
    for mode in d["modes"].values():
        assert list(generation_suffix(hybrid, mode["template_kwargs"])) == mode["generation_suffix_ids"]
    assert d["special_token_ids"]["</think>"] == 151668 and d["special_token_ids"]["<|im_end|>"] == 151645


def test_the_design_names_the_identity_and_the_frozen_task_lists():
    d, ident = load(DESIGN), load(IDENTITY)
    assert d["teacher"]["revision"] == ident["revision"]
    assert d["teacher"]["chat_template_sha256"] == ident["chat_template_sha256"]
    assert d["modes"]["H-T"]["think_state"] == ident["modes"]["thinking"]["think_state"]
    assert d["modes"]["H-N"]["think_state"] == ident["modes"]["non_thinking"]["think_state"]
    total = 0
    for row in d["tasks"]["T0"]["lists"]:
        p = REPO / row["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"]
        assert sum(1 for l in p.read_text().split("\n") if l.strip()) == row["tasks"]
        total += row["tasks"]
    assert total == d["tasks"]["T0"]["tasks"]


def test_the_preregistration_binds_the_rule_files_as_they_are():
    rule = load(PREREG)["decision_rule"]
    for path, sha in rule["bound_sha256"].items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == sha, path
    assert rule["constants"]["level"] == 0.975


def test_the_token_identity_audit_regenerates_identically():
    from stages.teacher_modes import token_identity_audit as T

    if not T.SESSIONS.is_file():
        pytest.skip("the H1-PA generation records are out of tree and absent here")
    try:
        doc = T.build()
    except OSError:
        pytest.skip("the Thinking-2507 tokenizer is not in the local HF cache")
    assert json.loads(json.dumps(doc)) == load(AUDIT)        # as written: JSON keys are strings
    iso = doc["as_generated_documents"]["isolation_check"]
    assert iso["pass"] and doc["matched_to_their_generation_turn"] == doc["sessions"]


def test_the_forecast_regenerates_identically():
    from stages.teacher_modes import forecast_cost

    if not (REPO / forecast_cost.RECORDS).is_dir():
        pytest.skip("the generation records are out of tree and absent here")
    assert forecast_cost.build() == load(FORECAST)

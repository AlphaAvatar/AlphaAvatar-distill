"""The 3 x 2 training plan's lineage and tokenizer binding (maintainer review 2026-11-01): every unit names a
tokenizer source that is a real, hash-pinned directory and the vocabulary hash the trainer requires; the lineage
binds the tokenizer's files and the initialization's search record; launch admission refuses another teacher's
tokenizer, another arm's pack, another lineage's initialization and an unstated RoPE base."""
from __future__ import annotations

import copy
import json

import pytest

from aadistill.training import lineage as LIN
from stages.teacher_modes import training_plan as TP

REPO = TP.REPO


@pytest.fixture(scope="module")
def units():
    adm = json.loads(TP.ADMISSION.read_text())
    return {u["unit"]: u for u in TP.units(adm, json.loads(TP.C1_AUDIT.read_text()))}


def test_every_unit_names_a_pinned_tokenizer_and_the_trainers_vocabulary_hash(units):
    ident = json.loads(TP.HYBRID.read_text())
    assert len(units) == 6
    for name, u in units.items():
        cfg = u["config"]
        assert cfg["tokenizer_sha256"] == ident["vocabulary_sha256"] == cfg["lineage"]["teacher"]["tokenizer_sha256"]
        key = "thinking_2507" if name.startswith("C1") else "hybrid"
        assert cfg["tokenizer_source"] == TP.TOKENIZERS[key]
        assert cfg["lineage"]["tokenizer"]["files_sha256"] == TP.tokenizer_files()[key]
        assert cfg["checkpoint"]["bind_identity"] is True
    # the hybrid's files are its snapshot's at the pinned revision; the two teachers' differ (the template)
    assert TP.tokenizer_files()["hybrid"] == {n: ident["files_sha256"][n] for n in TP.HYBRID_TOKENIZER_FILES}
    assert TP.tokenizer_files()["hybrid"]["tokenizer_config.json"] != \
        TP.tokenizer_files()["thinking_2507"]["tokenizer_config.json"]
    assert TP.validate(list(units.values())) == {n: {"problems": [], "pending_full_search": not n.startswith("C1")}
                                                for n in units}


def test_c1_binds_thinking_2507s_rope_and_its_d4_search_record(units):
    for seed in TP.SEEDS:
        lin = units[f"C1-s{seed}"]["config"]["lineage"]
        assert lin["teacher"]["rope_theta"] == 5_000_000.0 and lin["teacher"]["max_position_embeddings"] == 262144
        prov = lin["initialization"]["provenance"]
        assert prov["record"] == str(TP.D4_SEARCH.relative_to(REPO))
    a1 = units["A1-s20260726"]["config"]["lineage"]
    assert a1["teacher"]["rope_theta"] == 1_000_000.0 and a1["initialization"]["provenance"]["record"] == TP.UNSET


def _have(*paths):
    missing = [p for p in paths if not (REPO / p).exists()]
    if missing:
        pytest.skip(f"not on this machine: {missing}")


def test_c1_passes_launch_admission_on_the_committed_bytes(units):
    c1 = units["C1-s20260726"]["config"]
    _have(c1["student_path"], c1["data_dir"], c1["tokenizer_source"])
    rep = TP.admission_report([units["C1-s20260726"]])["C1-s20260726"]
    assert rep["problems"] == [] and {"initialization_provenance", "student_config", "student_weights_sha256",
                                      "tokenizer_files_sha256", "tokenizer_sha256"} <= set(rep["observed"])


def test_launch_admission_refuses_mixed_identities(units):
    c1 = units["C1-s20260726"]
    _have(c1["config"]["student_path"], c1["config"]["data_dir"], TP.TOKENIZERS["thinking_2507"],
          TP.TOKENIZERS["hybrid"], units["A2-s20260726"]["config"]["data_dir"])
    # the hybrid's tokenizer for C1: same vocabulary, another template
    bad = copy.deepcopy(c1)
    bad["config"]["tokenizer_source"] = TP.TOKENIZERS["hybrid"]
    assert any("tokenizer source's files differ" in p for p in TP.admission_report([bad])[c1["unit"]]["problems"])
    # A2's pack for C1
    bad = copy.deepcopy(c1)
    bad["config"]["data_dir"] = units["A2-s20260726"]["config"]["data_dir"]
    assert any("not a pack the lineage binds" in p for p in TP.admission_report([bad])[c1["unit"]]["problems"])
    # C1's Thinking-2507 initialization declared under the hybrid's lineage (A1's), its record and digest included
    bad = copy.deepcopy(units["A1-s20260726"])
    bad["config"]["student_path"] = c1["config"]["student_path"]
    bad["config"]["lineage"]["initialization"].update(
        model_safetensors_sha256=c1["config"]["lineage"]["initialization"]["model_safetensors_sha256"],
        provenance=c1["config"]["lineage"]["initialization"]["provenance"])
    probs = TP.admission_report([bad])[bad["unit"]]["problems"]
    assert any("recorded root Qwen/Qwen3-4B-Thinking-2507" in p for p in probs)
    assert any("rope_theta 5000000" in p for p in probs)
    with pytest.raises(LIN.LineageError):
        TP.admit([bad])


def test_a1_and_a2_are_refused_until_the_full_search_binds_their_initialization(units):
    _have(TP.TOKENIZERS["hybrid"])
    with pytest.raises(LIN.LineageError, match="full search's Top-1, which does not exist yet"):
        TP.admit([units["A1-s20260726"]])

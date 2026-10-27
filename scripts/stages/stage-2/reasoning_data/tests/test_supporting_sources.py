"""The supporting sources' checks accept their gold and refuse everything near
it; teacher code runs only in an isolated sandbox; a reused session whose bytes
changed is refused."""
from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_supporting_tasks as B
from stages.reasoning_data import supporting_verify as SV

from aadistill.data.target_quality import Check

POOLS = A.REPO / "data/stages/stage-2/reasoning_data/tasks"
MBPP = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--google-research-datasets--mbpp"
                                          "/snapshots/*/full/train-00000-of-00001.parquet")))
needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="no bubblewrap on this machine")


def test_boxed_math_accepts_the_gold_and_refuses_near_misses():
    t = {"answer": "\\frac{1}{2}"}
    assert SV.boxed_math(t, "", "So \\boxed{\\frac{1}{2}}.").state == "correct"
    assert SV.boxed_math(t, "", "So $\\boxed{ \\dfrac{1}{2} }$").state == "correct"
    assert SV.boxed_math(t, "", "\\boxed{\\frac{1}{3}}").state == "incorrect"
    assert SV.boxed_math(t, "", "\\boxed{0.5}").state == "incorrect"          # no symbolic equivalence
    no_box = SV.boxed_math(t, "\\boxed{\\frac{1}{2}}", "The answer is one half.")
    assert no_box.state == "incorrect" and "no boxed" in no_box.reason       # the reasoning is not read
    assert "no symbolic equivalence" in no_box.scope


def test_the_reuse_keys_are_verify_py_rules():
    num = {"answer": "48/2 = 24\n48+24 = 72", "group": "code_math", "corpus_source": "gsm8k"}
    assert SV.VERIFIERS["final_number"](num, "", "So she sold 72 clips.").state == "correct"
    assert SV.VERIFIERS["final_number"](num, "", "So she sold 71 clips.").state == "incorrect"
    span = {"answer": "no", "group": "multihop_qa", "corpus_source": "hotpot_qa"}
    assert SV.VERIFIERS["span"](span, "", "No, they are not.").state == "correct"
    assert SV.VERIFIERS["span"](span, "", "Yes -- there is no doubt.").state == "incorrect"


ABSTAIN = ["The provided context does not contain the information needed to answer this question.",
           "I can't answer that from the given context — the passage doesn't mention it.",
           "This isn't stated in the provided context, so I can't give a grounded answer.",
           "I cannot answer from the context.",
           "**Answer:** The context does not mention when the treaty was signed.",
           "The information about the director is not provided in the given paragraphs."]
NOT_ABSTAIN = ["Based on the provided context, she rose to fame in the late 1990s. The context does not "
               "specify the exact year.",                                   # an answer with a caveat
               "I'm not sure.",                                             # names no context
               "I cannot answer from the context. However, based on general knowledge, it is 1990.",
               "The context does not state it, but the answer is 1990.",
               "The context does not mention it. \\boxed{1990}",
               "The war ended in 1945.",
               ""]


@pytest.mark.parametrize("text", ABSTAIN)
def test_explicit_abstentions_are_accepted(text):
    assert SV.abstention({}, "", text).state == "correct", SV.is_abstention(text)


@pytest.mark.parametrize("text", NOT_ABSTAIN)
def test_answers_caveats_and_vague_replies_are_not_abstentions(text):
    c = SV.abstention({}, "", text)
    assert c.state == "incorrect" and "no length rule" in c.scope


def test_the_abstention_check_has_no_length_rule():
    long = "I cannot answer from the context. " + "The passage discusses the city's history at length. " * 40
    assert SV.abstention({}, "", long).state == "correct"


def test_the_stage2_v0_unanswerable_golds_are_abstentions():
    from shared.data.build_stage2_v0 import UNANSWERABLE_RESPONSES
    assert all(SV.is_abstention(r)[0] for r in UNANSWERABLE_RESPONSES)


def test_code_block_takes_the_last_python_block():
    final = "Here:\n```python\ndef f():\n    return 1\n```\nand\n```\ndef f():\n    return 2\n```\n"
    code, n = SV.code_block(final)
    assert n == 2 and "return 2" in code
    assert SV.code_block("def f(): return 1")[0] is None


def _mbpp(n=3):
    import pyarrow.parquet as pq
    return [r for r in pq.read_table(MBPP[-1]).to_pylist()][:n]


@needs_bwrap
@pytest.mark.skipif(not MBPP, reason="MBPP not in the local HF cache")
def test_mbpp_reference_code_passes_and_a_mutant_fails_in_the_sandbox():
    for r in _mbpp():
        task = {"test_list": r["test_list"], "test_setup_code": r["test_setup_code"]}
        good = SV.code_asserts(task, "", f"```python\n{r['code']}\n```")
        assert good.state == "correct", good.reason
        bad = SV.code_asserts(task, "", "```python\n" + r["code"].replace("return", "return None and", 1) + "\n```")
        assert bad.state == "incorrect"
        assert SV.code_asserts(task, "", r["code"]).state == "incorrect"      # no fenced block
    assert "not full semantic correctness" in good.scope


def test_no_teacher_code_runs_when_the_sandbox_is_not_isolated(monkeypatch):
    from stages.reasoning_data import code_sandbox as CS
    monkeypatch.setattr(SV, "_sandbox", None)
    monkeypatch.setattr(CS, "bwrap_sandbox", lambda *a, **k: object())
    monkeypatch.setattr(CS, "probe_sandbox", lambda *a, **k: {"isolated": False, "reached": ["network"]})

    def ran(*a, **k):
        raise AssertionError("teacher code ran without an isolated sandbox")
    monkeypatch.setattr(SV, "run_asserts", ran)
    with pytest.raises(SystemExit, match="not isolated"):
        SV.code_asserts({"test_list": ["assert f() == 1"]}, "", "```python\ndef f(): return 1\n```")


def _session(i, answer="72"):
    return {"id": f"gsm8k-{i:06d}#t1", "source_id": f"gsm8k-{i:06d}", "data_type": "gsm8k",
            "candidate_index": 0, "candidate_sha256": "x",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"q{i}"},
                         {"role": "assistant", "content": answer, "reasoning_content": "r"}], "tools": None}


def _selection(tmp_path, sessions):
    rows = [{"session_id": s["id"], "unit_id": s["source_id"], "unit_sessions": 1,
             "session_sha256": B.canonical_sha(s), "stratum": "supporting.math_reused_gsm8k",
             "source": "corpus_v2_gsm8k_verified", "candidate_index": 0, "verdict": "correct"}
            for s in sessions]
    p = tmp_path / "sel.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_the_reuse_reader_returns_the_packer_contract(tmp_path, monkeypatch):
    ss = [_session(0), _session(1)]
    sel = _selection(tmp_path, ss)
    monkeypatch.setattr(A, "corpus_v2_sessions", lambda: iter(ss))
    out = B.reused_sessions(sel)
    assert [o["id"] for o in out] == [s["id"] for s in ss]
    assert all(o["unit_sessions"] == 1 and o["stratum"] == o["data_type"] and o["messages"] for o in out)


def test_the_reuse_reader_refuses_a_tampered_or_missing_session(tmp_path, monkeypatch):
    ss = [_session(0), _session(1)]
    sel = _selection(tmp_path, ss)
    monkeypatch.setattr(A, "corpus_v2_sessions", lambda: iter([_session(0), _session(1, answer="71")]))
    with pytest.raises(ValueError, match="differ"):
        B.reused_sessions(sel)
    monkeypatch.setattr(A, "corpus_v2_sessions", lambda: iter([_session(0)]))
    with pytest.raises(ValueError, match="missing"):
        B.reused_sessions(sel)


POOL_FILES = {"math_train": "math_train_v1.jsonl", "mbpp_train": "mbpp_train_v1.jsonl",
              "squad_v2_train_unanswerable": "squad_v2_train_unanswerable_v1.jsonl"}


@pytest.mark.parametrize("source", sorted(POOL_FILES))
def test_every_built_task_carries_its_provenance_and_a_working_check(source):
    path = POOLS / POOL_FILES[source]
    if not path.exists():
        pytest.skip("pool not built")
    man = A.read_json(path.with_name(path.stem + ".manifest.json"))
    assert man["sha256"] == A.sha256_file(path) and man["isolation_pass"] is True
    tasks = list(A.jsonl(path))
    assert len(tasks) == man["n_tasks"] >= man["demand"]["tasks"]
    for t in tasks:
        assert t["source"] == source and t["source_split"] and t["family_id"] and t["unit_id"] == t["task_id"]
        assert t["verifier"] in SV.VERIFIERS and t["interactive"] is False
        if source == "math_train":
            assert SV.boxed_math(t, "", f"\\boxed{{{t['answer']}}}").state == "correct"
            assert SV.boxed_math(t, "", "\\boxed{zz9q}").state == "incorrect"
        if source == "mbpp_train":
            assert t["test_list"] and all(a in t["messages"][0]["content"] for a in t["test_list"])
        if source == "squad_v2_train_unanswerable":
            assert t["answer"] == "" and "say you cannot answer from the context" in t["messages"][0]["content"]


def test_the_reuse_selection_holds_whole_verified_units():
    path = POOLS / "supporting_reuse_v1.selection.jsonl"
    if not path.exists():
        pytest.skip("selection not built")
    rows = list(A.jsonl(path))
    assert all(r["verdict"] == "correct" and r["unit_sessions"] == 1 for r in rows)
    assert len({r["unit_id"] for r in rows}) == len(rows)
    assert {r["source"] for r in rows} == {"corpus_v2_gsm8k_verified", "corpus_v2_rag_verified",
                                           "corpus_v2_multihop_verified"}
    if (A.CORPUS_V2 / "sessions.jsonl").exists():
        got = B.reused_sessions(path)                     # every session re-identified from corpus_v2
        assert [g["id"] for g in got] == [r["session_id"] for r in rows]


def test_a_verifier_returns_a_check():
    assert isinstance(SV.abstention({}, "", "I cannot answer from the context."), Check)

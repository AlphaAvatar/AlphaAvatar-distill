"""The three think states score one answer identically, and `preclosed` is opt-in.

Every registered eval-set-1 conformance vector whose reply has one well-formed
reasoning block is rewritten into the two hybrid forms -- thinking
(``self_opening``: the model opens ``<think>`` itself) and non-thinking
(``preclosed``: the prompt closed an empty block, the reply is the answer) --
and must receive exactly the historical verdict. A record that does not state
``think_state`` keeps the historical rule, so the same non-thinking reply
without the field is a protocol violation, as it always was.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
EVAL_IDENTITY = REPO / "configs/stages/stage-2/teacher_modes/evaluation_identity.json"
TEACHER_IDENTITY = REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json"


def forms(raw: str) -> dict | None:
    """The thinking and non-thinking spellings of a historical (pre-opened) reply."""
    if raw.count("</think>") != 1 or "<think>" in raw:
        return None
    _, _, after = raw.partition("</think>")
    return {"self_opening": "<think>" + raw, "preclosed": after.lstrip("\n")}


def rewrite(record: dict, state: str) -> dict | None:
    r = copy.deepcopy(record)
    targets = r["turns"] if "turns" in r else [r]
    for t in targets:
        f = forms(t.get("raw", ""))
        if f is None:
            return None
        t["raw"], t["think_state"] = f[state], state
    return r


def vectors():
    from shared import eval_sets as ES
    from shared.evaluation import eval_set_scoring as S

    eid = ES.default_formal()
    m, root = ES.manifest(eid), ES.set_root(eid)
    for cname, comp in m["components"].items():
        index = {u: {r[s["id_field"]]: r for r in ES.items(eid, cname, u)}
                 for u, s in comp["units"].items() if not s.get("withheld")}
        for protocol, rel in S.conformance_files(comp["scorer"]).items():
            for line in (root / rel).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    yield cname, protocol, index[r["unit"]][r["id"]], r


def test_every_rewritable_conformance_vector_scores_the_same_in_all_three_states():
    from shared.evaluation import eval_set_scoring as S

    compared = {"self_opening": 0, "preclosed": 0}
    for cname, protocol, item, r in vectors():
        base = S.ADAPTERS[cname](r["unit"], item, r["input"], protocol)
        assert base == r["expected"], (cname, r["unit"], r["id"], r["variant"])   # history unchanged
        for state in compared:
            rec = rewrite(r["input"], state)
            if rec is None:
                continue
            got = S.ADAPTERS[cname](r["unit"], item, rec, protocol)
            assert got == base, (cname, protocol, r["unit"], r["id"], r["variant"], state, base, got)
            compared[state] += 1
    # every component and both protocols contribute (13,372 comparisons at writing)
    assert min(compared.values()) > 6000


def test_preclosed_is_opt_in_and_refuses_any_think_delimiter():
    from aadistill.evaluation.behavior import record_think_state, split_generation
    from aadistill.evaluation.strict_answer import protocol_valid

    reply = "The answer is 5.<|im_end|>"
    assert record_think_state({}) == "preopened"                      # a record without the field
    assert record_think_state({"think_preopened": False}) == "self_opening"
    assert record_think_state({"think_state": "preclosed"}) == "preclosed"
    with pytest.raises(ValueError):
        record_think_state({"think_state": "closed"})
    valid = lambda raw, **kw: protocol_valid(raw, **kw)[0]  # noqa: E731 -- (ok, reason)
    assert valid(reply, think_state="preclosed")
    assert protocol_valid(reply) == (False, "think_delimiters_invalid")   # historical rule: no </think>
    for bad in ("<think>r</think>5<|im_end|>", "r</think>5<|im_end|>", "<think>5<|im_end|>"):
        assert not valid(bad, think_state="preclosed"), bad
    assert split_generation(reply, think_state="preclosed")["answer"] == "The answer is 5."
    # the thinking mode keeps its own rule: the model must open the block itself
    assert valid("<think>r</think>\n\n5<|im_end|>", think_state="self_opening")
    assert not valid("r</think>\n\n5<|im_end|>", think_state="self_opening")


def test_the_evaluation_identity_regenerates_and_binds_each_mode():
    from stages.teacher_modes import evaluation_identity as E

    doc = E.build()
    assert doc == json.loads(EVAL_IDENTITY.read_text())
    ident = json.loads(TEACHER_IDENTITY.read_text())
    ht, hn = doc["modes"]["H-T"], doc["modes"]["H-N"]
    assert (ht["think_state"], hn["think_state"]) == ("self_opening", "preclosed")
    assert ht["chat_template_sha256"] == hn["chat_template_sha256"] == ident["chat_template_sha256"]
    assert ht["identity_sha256"] != hn["identity_sha256"]
    assert ht["eval_set"] == hn["eval_set"] and ht["decoding"] == hn["decoding"]   # one instrument
    for path, digest in ht["scorer"]["sources_sha256"].items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == digest, path

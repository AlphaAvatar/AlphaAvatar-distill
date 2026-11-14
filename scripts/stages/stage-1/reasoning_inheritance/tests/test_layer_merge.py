"""The layer-merge round (RI-LM1): its design's pins, its decision rule, and the pod pipeline on the tiny real Qwen3.

The pod runs `layer_merge.collect_statistics -> alignments -> equivalence_checks -> build_candidate -> diagnostics` on
the 4B teacher; every one of them runs here end to end on the toy teacher (six blocks, spans (1, 2) and (3, 4)), so a
defect in the pipeline's own logic is found at $0 rather than on a pod. CUDA numerics and the real model's memory are
the pod's to show.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reasoning_inheritance import layer_merge as LMX
from reasoning_inheritance import layer_merge_analysis as LA
from aadistill.initialization.planning import frozen_mapping as FM
from aadistill.initialization.specs.arch import get_adapter

REPO = Path(__file__).resolve().parents[5]
ADAPTER = get_adapter("qwen3")


# ---- the design's pins ----------------------------------------------------------------------------------------------

def test_the_spans_are_the_frozen_selections_and_the_protocol_is_ri_s1s():
    d = LMX.design()
    assert LMX.merge_spans(d) == [(2, 3), (10, 11), (16, 17, 18), (24, 25, 26), (29, 30), (31, 32)]
    ri = json.loads((REPO / d["base"]["ri_s1_design"]["path"]).read_text())
    got = hashlib.sha256(json.dumps(ri["evaluation"], sort_keys=True).encode()).hexdigest()
    assert got == d["evaluation"]["protocol_sha256"]
    for k in ("ri_s1_design", "frozen_selection"):
        assert hashlib.sha256((REPO / d["base"][k]["path"]).read_bytes()).hexdigest() == d["base"][k]["sha256"]
    dig = json.loads(LMX.DIGESTS.read_text())
    assert dig["checkpoint"]["weights_sha256"] == d["base"]["d_checkpoint"]["weights_sha256"]
    assert dig["n_tensors"] == 28 * 11 + 2                   #: 28 blocks x 11 tensors, the embedding, the final norm


def test_the_evaluation_order_measures_the_primary_unit_of_both_candidates_first():
    order = [tuple(x) for x in LMX.design()["evaluation"]["order"]]
    assert order[:2] == [("M-avg", "H-T", "fluid"), ("M-act", "H-T", "fluid")]
    assert sorted(order) == sorted((g, m, k) for g in LMX.CANDIDATES for m in ("H-T", "H-N")
                                   for k in ("fluid", "capability"))


# ---- the decision rule ----------------------------------------------------------------------------------------------

def test_the_sign_flip_test_and_holm_are_the_registered_definitions():
    zero = LA.sign_flip_test(np.zeros(10, dtype=np.int64), draws=1000)
    assert zero["p_two_sided"] == 1.0                         #: |sum| >= 0 always: no evidence, p = 1
    strong = LA.sign_flip_test(np.ones(40, dtype=np.int64), draws=2000)
    assert strong["p_two_sided"] == pytest.approx(1 / 2001)   #: no flip reaches |40|... but all +1 or all -1 would
    again = LA.sign_flip_test(np.ones(40, dtype=np.int64), draws=2000)
    assert again == strong                                    #: seeded: the same p every time
    assert LA.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert LA.holm({"a": 0.03, "b": 0.02}) == {"b": 0.04, "a": 0.04}   #: monotone in the sorted order
    assert LA.verdict(0.05, 0.01) == "BETTER" and LA.verdict(-0.05, 0.01) == "WORSE"
    assert LA.verdict(0.05, 0.2) == "NOT_DISTINGUISHED"


def test_decide_pairs_by_group_reports_the_interval_apart_and_refuses_an_incomplete_unit():
    groups = {f"g{i}": [f"i{2 * i}", f"i{2 * i + 1}"] if i < 3 else [f"i{100 + i}"] for i in range(60)}
    items = [i for v in groups.values() for i in v]
    y_d = {i: 0 for i in items}
    better = {i: (1 if k % 3 == 0 else 0) for k, i in enumerate(items)}
    out = LA.decide({"M-avg": better, "M-act": None}, y_d, groups)
    assert out["M-act"]["verdict"] == "not_measured"
    m = out["M-avg"]
    assert m["n_groups"] == 60 and m["n_items"] == len(items) == 63
    assert m["delta"] == pytest.approx(sum(better.values()) / 63)
    assert m["verdict"] == "BETTER" and m["p_holm"] == m["test"]["p_two_sided"]   #: one measured: Holm over one
    assert m["effect_interval"]["label"].startswith("effect interval, unadjusted")
    lo, hi = m["effect_interval"]["ci95_percentile"]
    assert lo <= m["delta"] <= hi
    with pytest.raises(ValueError):
        LA.group_differences({"i0": 1}, y_d, groups)


# ---- the pipeline on the toy teacher --------------------------------------------------------------------------------

def _batches(n=3, t=40, vocab=128):
    g = torch.Generator().manual_seed(5)
    seqs = [torch.randint(0, vocab, (1, t), generator=g) for _ in range(n)]

    def make():
        for s in seqs:
            yield SimpleNamespace(batch=SimpleNamespace(input_ids=s, attention_mask=torch.ones_like(s)),
                                  original_indices=(0,))
    return make


@torch.no_grad()
def test_the_pipeline_builds_both_candidates_on_the_toy_teacher(teacher):
    sel = {"n_root_blocks": 6, "kept_blocks": [0, 1, 3, 5]}
    spans = [s for s in LMX.LM.spans_from_kept_blocks(6, sel["kept_blocks"]) if len(s) > 1]
    assert spans == [(1, 2), (3, 4)]
    n_q, n_kv, _ = ADAPTER.head_groups(ADAPTER.spec_of(teacher))
    batches = _batches()
    st = LMX.collect_statistics(ADAPTER, teacher, spans, batches, None, chunk=16)
    assert st.sequences == 3 and st.positions == 120
    al = LMX.alignments(st, n_q, n_kv)
    assert set(al) == {(1, 2), (3, 4)}
    for x in al.values():
        assert sorted(x["ffn_perm"].tolist()) == list(range(48))
        assert x["summary"]["ffn_matched_correlation_mean"] >= x["summary"]["ffn_identity_correlation_mean"] - 1e-12
    stats = LMX.statistics_tensors(st, al)
    eq = LMX.equivalence_checks(ADAPTER, teacher, spans, stats, next(iter(batches())), 1e-4)
    assert eq["passed"] and set(eq["blocks"]) == {"1", "2", "3", "4"} and "permutation" in eq["blocks"]["2"]
    d = FM.keep_blocks(ADAPTER, teacher, sel["kept_blocks"])
    digests = {n: FM.tensor_sha256(t) for n, t in d.state_dict().items() if n != "lm_head.weight"}
    built = {}
    for method in LMX.CANDIDATES:
        model, rec = LMX.build_candidate(ADAPTER, teacher, sel, spans, method, stats, digests, seed=0)
        assert rec["checks"]["tensors_outside_spans_differ"] == [] and all(rec["checks"]["span_blocks_differ_from_D"].values())
        assert rec["checks"]["num_parameters"] == sum(p.numel() for p in d.parameters())
        built[method] = model
    #: M-avg's span block is the plain average of the span's teacher tensors (norm gains included)
    blocks = ADAPTER.blocks(teacher)
    avg = ADAPTER.blocks(built["M-avg"])[1]
    want = (blocks[1].mlp.gate_proj.weight.double() + blocks[2].mlp.gate_proj.weight.double()) / 2
    assert torch.allclose(avg.mlp.gate_proj.weight.double(), want.to(avg.mlp.gate_proj.weight.dtype).double())
    assert torch.allclose(avg.input_layernorm.weight, (blocks[1].input_layernorm.weight + blocks[2].input_layernorm.weight) / 2)
    #: M-act's gains are folded: its norms are one
    act = ADAPTER.blocks(built["M-act"])[1]
    assert torch.equal(act.input_layernorm.weight, torch.ones_like(act.input_layernorm.weight))
    #: a candidate that is not D outside the spans is refused
    bad = dict(digests)
    bad["model.embed_tokens.weight"] = "0" * 64
    with pytest.raises(LMX.MergeError):
        LMX.build_candidate(ADAPTER, teacher, sel, spans, "M-avg", stats, bad, seed=0)
    import copy
    cands = {"D": {s[0]: copy.deepcopy(blocks[s[0]]) for s in spans}}
    for m, model in built.items():
        cands[m] = {s[0]: copy.deepcopy(ADAPTER.blocks(model)[LMX.child_index(sel, s[0])]) for s in spans}
    diag = LMX.diagnostics(ADAPTER, teacher, spans, cands, batches, None)
    assert diag["positions"] == 120
    for s in ("[1, 2]", "[3, 4]"):
        for m in ("D", *LMX.CANDIDATES):
            for c in ("attention", "ffn", "block"):
                r = diag["spans"][s][m][c]
                assert r["positions"] == 120 and r["normalized_error"] is not None and r["rms_reference"] > 0

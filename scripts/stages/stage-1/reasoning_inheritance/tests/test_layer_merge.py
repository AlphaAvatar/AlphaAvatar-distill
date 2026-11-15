"""The layer-merge round (RI-LM1): its design's pins, its decision rule, the pod pipeline on the tiny real Qwen3, and
what makes a failed attempt cheap.

The pod runs `layer_merge.collect_statistics -> alignments -> equivalence_checks -> build_candidate -> diagnostics` on
the 4B teacher; every one of them runs here end to end on the toy teacher (six blocks, spans (1, 2) and (3, 4)), so a
defect in the pipeline's own logic is found at $0 rather than on a pod. Beside them: Holm over both registered
comparisons; a candidate as D plus its span-block delta, byte for byte; reuse only under a product's identity; a
capability unit sealed the moment it is written, with a stand-in runner that keeps the real runner's write order
(checked against its source); the dev-box retention of the deltas. CUDA numerics and the real model's memory are the
pod's to show.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reasoning_inheritance import layer_merge as LMX
from reasoning_inheritance import layer_merge_analysis as LA
from stages.reasoning_inheritance import layer_merge_driver as LMD
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


def _toy_groups():
    groups = {f"g{i}": [f"i{2 * i}", f"i{2 * i + 1}"] if i < 3 else [f"i{100 + i}"] for i in range(60)}
    items = [i for v in groups.values() for i in v]
    return groups, items


def test_holm_runs_over_both_registered_comparisons_or_there_is_no_verdict():
    """The family is the two registered comparisons, always: a candidate that is not measured never shrinks Holm's m
    to one, and while either primary unit is incomplete neither candidate gets a test or a verdict."""
    groups, items = _toy_groups()
    y_d = {i: 0 for i in items}
    better = {i: (1 if k % 3 == 0 else 0) for k, i in enumerate(items)}
    out = LA.decide({"M-avg": better, "M-act": None}, y_d, groups)
    assert out["_family"] == {"comparisons": ["M-avg - D", "M-act - D"], "holm_m": 2,
                              "complete": {"M-avg": True, "M-act": False}, "formal": False,
                              "status": "descriptive_only"}
    assert out["M-act"]["verdict"] == "not_measured" and out["M-act"]["items_present"] == 0
    m = out["M-avg"]
    assert m["verdict"] == "descriptive_only" and "test" not in m and "p_holm" not in m
    assert m["n_groups"] == 60 and m["n_items"] == len(items) == 63
    assert m["delta"] == pytest.approx(sum(better.values()) / 63)
    assert m["effect_interval"]["label"].startswith("effect interval, unadjusted")
    lo, hi = m["effect_interval"]["ci95_percentile"]
    assert lo <= m["delta"] <= hi
    #: one registered item without a record leaves the unit incomplete
    part = dict(y_d)
    part.pop(items[5])
    out = LA.decide({"M-avg": better, "M-act": part}, y_d, groups)
    assert out["M-act"]["items_present"] == 62 and out["_family"]["status"] == "descriptive_only"
    #: both complete: tests, and Holm over two
    out = LA.decide({"M-avg": better, "M-act": dict(y_d)}, y_d, groups)
    assert out["_family"]["status"] == "formal"
    pa, pb = out["M-avg"]["test"]["p_two_sided"], out["M-act"]["test"]["p_two_sided"]
    assert pb == 1.0 and out["M-avg"]["p_holm"] == pytest.approx(min(1.0, 2 * pa)) and out["M-act"]["p_holm"] == 1.0
    assert out["M-avg"]["verdict"] == "BETTER" and out["M-act"]["verdict"] == "NOT_DISTINGUISHED"
    with pytest.raises(ValueError):
        LA.decide({"M-avg": better, "M-act": better}, {"i0": 0}, groups)    #: D must cover every registered item
    with pytest.raises(ValueError):
        LA.group_differences({"i0": 1}, y_d, groups)


def test_the_primary_items_are_the_registered_570_in_558_groups():
    ids = LA.registered_primary_items()
    assert len(ids) == 570 and len(LA.independence_groups(ids)) == 558


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


def test_the_teardown_reserve_and_the_budget_terminal_state_refuse():
    from reasoning_inheritance import layer_merge_launch as L
    assert L.relay_object_sizes() == [1_211_169_792 + 2**20] * 2 + [64 * 2**20]       #: two deltas, the statistics
    g = {"teardown_reserve_usd": 1.0, "min_useful_attempt_usd": 2.0}
    env, pos = {"remaining_usd": 13.0}, {"project_remaining_usd": 36.76, "balance_usd": 16.63}
    assert L.reserve_problems(11.0, g, env, pos) == []
    assert "budget is exhausted" in L.reserve_problems(1.5, g, env, pos)[0]           #: below the smallest attempt
    assert L.reserve_problems(11.0, g, {"remaining_usd": 10.0}, pos)                  #: the ceiling's remainder
    assert L.reserve_problems(11.0, g, env, {"project_remaining_usd": 11.5, "balance_usd": 50})   #: cap + reserve
    assert L.reserve_problems(11.0, g, env, {"project_remaining_usd": 50, "balance_usd": 11.5})   #: the balance
    proposal = json.loads((REPO / L.STUDY.proposal).read_text())
    assert proposal["status"] == "proposed" and proposal["study"] == L.STUDY.key
    assert proposal["ceiling_usd"] + proposal["teardown_reserve_usd"] == proposal["study_cumulative_cap_usd"]
    assert proposal["per_session_hard_cap_usd"] <= proposal["ceiling_usd"]
    assert not (REPO / L.STUDY.grant).exists()                                         #: proposed, never granted here


# ---- a candidate as D plus its span blocks; the checks refuse non-finite errors ----------------------------------------

@torch.no_grad()
def test_d_plus_the_delta_is_the_candidate_byte_for_byte(teacher, tmp_path):
    """The lossless storage: D (keep_blocks of the teacher) with the delta FILE's tensors written in is the candidate,
    tensor for tensor and as the saved artifact (the pod verifies exactly this before a candidate counts as built)."""
    from safetensors.torch import load_file, save_file
    RI = LMD.RI
    sel = {"n_root_blocks": 6, "kept_blocks": [0, 1, 3, 5]}
    spans = [s for s in LMX.LM.spans_from_kept_blocks(6, sel["kept_blocks"]) if len(s) > 1]
    d = FM.keep_blocks(ADAPTER, teacher, sel["kept_blocks"])
    digests = {n: FM.tensor_sha256(t) for n, t in d.state_dict().items() if n != "lm_head.weight"}
    model, _ = LMX.build_candidate(ADAPTER, teacher, sel, spans, "M-avg", None, digests, seed=0)
    delta = LMX.candidate_delta(sel, spans, model)
    assert sorted({int(n.split(".")[2]) for n in delta}) == [1, 2] and len(delta) == 2 * 11
    save_file(delta, str(tmp_path / "delta.safetensors"))
    base = FM.keep_blocks(ADAPTER, teacher, sel["kept_blocks"])
    LMX.apply_delta(base, load_file(str(tmp_path / "delta.safetensors")))
    want = {n: FM.tensor_sha256(t) for n, t in model.state_dict().items()}
    assert {n: FM.tensor_sha256(t) for n, t in base.state_dict().items()} == want
    a, b = RI._save(ADAPTER, model, tmp_path / "cand"), RI._save(ADAPTER, base, tmp_path / "rebuilt")
    assert a.artifact_digest == b.artifact_digest and a.shards[0].sha256 == b.shards[0].sha256
    bad = dict(delta)
    k = next(iter(bad))
    bad[k] = bad[k].to(torch.float64)
    with pytest.raises(LMX.MergeError, match="does not fit D"):
        LMX.apply_delta(FM.keep_blocks(ADAPTER, teacher, sel["kept_blocks"]), bad)


@torch.no_grad()
def test_a_non_finite_equivalence_error_fails_the_check(teacher):
    import copy
    sel_kept = [0, 1, 3, 5]
    spans = [s for s in LMX.LM.spans_from_kept_blocks(6, sel_kept) if len(s) > 1]
    n_q, n_kv, _ = ADAPTER.head_groups(ADAPTER.spec_of(teacher))
    batches = _batches()
    st = LMX.collect_statistics(ADAPTER, teacher, spans, batches, None, chunk=16)
    stats = LMX.statistics_tensors(st, LMX.alignments(st, n_q, n_kv))
    broken = copy.deepcopy(teacher)
    ADAPTER.blocks(broken)[2].mlp.down_proj.weight[0, 0] = float("nan")
    eq = LMX.equivalence_checks(ADAPTER, broken, spans, stats, next(iter(batches())), 1e-4)
    assert not eq["passed"] and eq["non_finite"] and all(x.startswith(("2:", "3:", "4:")) for x in eq["non_finite"])


def test_the_code_identity_ignores_comments_and_docstrings_but_not_statements(tmp_path, monkeypatch):
    import importlib
    import sys as _sys
    bodies = {"a": 'def f(x):\n    """one."""\n    return x + 1\n',
              "b": '# a comment\n\n\ndef f(x):\n    """another docstring."""\n    # inside\n    return x + 1\n',
              "c": 'def f(x):\n    return x + 2\n'}
    monkeypatch.syspath_prepend(str(tmp_path))
    got = {}
    for k, body in bodies.items():
        (tmp_path / f"cid_{k}.py").write_text(body)
    importlib.invalidate_caches()
    for k in bodies:
        mod = importlib.import_module(f"cid_{k}")
        got[k] = next(iter(LMX.code_identity(mod.f)["objects"].values()))
        _sys.modules.pop(f"cid_{k}", None)
    assert got["a"] == got["b"] != got["c"]


# ---- the driver reuses a product only under this attempt's identity ----------------------------------------------------

def _ids(out: Path, cand_sha: dict) -> None:
    LMD.RI.write_json(out / "state" / "identities.json",
                      {"candidates": {g: {"kind": "candidate", "sha256": h} for g, h in cand_sha.items()}})


def test_a_candidate_is_reused_only_under_this_attempts_identity(tmp_path):
    RI = LMD.RI
    out = tmp_path / "out"
    _ids(out, {"M-avg": "a" * 64, "M-act": "b" * 64})
    dp = out / "products" / "M-avg" / "delta.safetensors"
    dp.parent.mkdir(parents=True)
    dp.write_bytes(b"delta")
    rec = {"identity": {"sha256": "a" * 64}, "status": "durable", "relay": {"verified": True},
           "delta": {"path": "products/M-avg/delta.safetensors", "sha256": RI.sha256_file(dp)}}
    RI.write_json(out / "records" / "candidates" / "M-avg.json", rec)
    assert LMD.candidate_state(out, "M-avg", "att2") == "durable"
    RI.write_json(out / "records" / "candidates" / "M-avg.json", {**rec, "status": "built", "relay": None})
    assert LMD.candidate_state(out, "M-avg", "att2") == "built"           #: only the relay step is left
    #: another identity (say the merge code changed): superseded with the reason, never used
    RI.write_json(out / "records" / "candidates" / "M-avg.json", {**rec, "identity": {"sha256": "c" * 64}})
    assert LMD.candidate_state(out, "M-avg", "att3") == "absent"
    assert not (out / "records" / "candidates" / "M-avg.json").exists() and not dp.exists()
    moved = out / "superseded" / "att3"
    assert (moved / "records" / "candidates" / "M-avg.json").is_file() and (moved / "products" / "M-avg").is_dir()
    assert "is not this attempt's" in RI.read_json(moved / "why.json")[0]["why"]
    #: built, but the delta is gone (the next pod's evidence directory starts without it): rebuilt, not trusted
    RI.write_json(out / "records" / "candidates" / "M-act.json",
                  {"identity": {"sha256": "b" * 64}, "status": "built",
                   "delta": {"path": "products/M-act/delta.safetensors", "sha256": "0" * 64}})
    assert LMD.candidate_state(out, "M-act", "att4") == "absent"


def test_sealed_evaluation_work_is_taken_only_under_its_identity(tmp_path):
    out = tmp_path / "out"
    unit = out / "eval" / "M-avg_H-T"
    (unit / "parts").mkdir(parents=True)
    (unit / "parts" / "part_0000.jsonl").write_text('{"task_id": "x"}\n')
    want = LMD.identity("evaluation_unit", candidate="d1", mode="H-T")
    assert LMD.admit_unit(out, unit, want, "att1") == "superseded"      #: sealed work with no identity
    assert not (unit / "parts").exists() and (out / "superseded" / "att1" / "eval" / "M-avg_H-T" / "parts").is_dir()
    (unit / "parts").mkdir()
    (unit / "parts" / "part_0000.jsonl").write_text('{"task_id": "y"}\n')
    assert LMD.admit_unit(out, unit, want, "att2") == "taken"
    assert (unit / "parts" / "part_0000.jsonl").is_file()
    other = LMD.identity("evaluation_unit", candidate="d2", mode="H-T")
    assert LMD.admit_unit(out, unit, other, "att3") == "superseded"


def test_the_identities_are_computed_from_the_design_session_and_code(tmp_path):
    ids = LMD.identities(tmp_path / "work", "cpu")
    st, cand = ids["statistics"], ids["candidates"]
    assert st["parts"]["data"]["positions"]["policy"].startswith("positions.supervised_target_v1")
    assert cand["M-act"]["parts"]["statistics"] == st["sha256"] and "statistics" not in cand["M-avg"]["parts"]
    assert ids["evaluation"]["parts"]["protocol_sha256"] == LMX.design()["evaluation"]["protocol_sha256"]


# ---- capability units: sealed the moment they are written; a later attempt generates only what is missing ----------

FAKE_RUNNER = r"""#!{python}
import json, os, sys, time
from pathlib import Path
sys.path[:0] = ["src", "scripts"]
from shared.evaluation import decoding as DEC
a = sys.argv[2:]
def val(f):
    return a[a.index(f) + 1]
prompts = a[a.index("--prompts") + 1:a.index("--out-dir")]
raw, stream = Path(val("--out-dir")), Path(val("--stream-completed"))
proto = DEC.load(val("--decoding"))
raw.mkdir(parents=True, exist_ok=True)
stream.mkdir(parents=True, exist_ok=True)
log = Path(os.environ["FAKE_LOG"])
fail_after = int(os.environ.get("FAKE_FAIL_AFTER", "-1"))
done = 0
for pth in prompts:
    stem = Path(pth).stem
    ids = [json.loads(l)["id"] for l in Path(pth).read_text().split("\n") if l.strip()]
    with log.open("a") as fh:
        fh.write(json.dumps({"unit": stem, "ids": ids}) + "\n")
    recs = []
    for i in sorted(ids):
        r = {"id": i, "label": "x", "raw": "ok", "stop_reason": "eos", "sampling_seed": DEC.request_seed(proto, stem, i)}
        recs.append(r)
        with open(stream / f"{stem}.stream.jsonl", "a") as fh:
            fh.write(json.dumps(r) + "\n")
        done += 1
        if done == fail_after:
            sys.exit(3)
    g = raw / f"{stem}.generations.jsonl"
    g.with_name("." + g.name).write_text("".join(json.dumps(r) + "\n" for r in recs)); os.replace(g.with_name("." + g.name), g)
    s = raw / f"{stem}.json"
    s.with_name("." + s.name).write_text(json.dumps({"n_samples": len(recs)})); os.replace(s.with_name("." + s.name), s)
    #: the driver must seal this unit before the runner starts the next one
    gens = raw.parent.parent / "eval" / raw.name.split(".")[0] / "generations" / f"{stem}.jsonl"
    t0 = time.time()
    while not gens.is_file() and time.time() - t0 < 20:
        time.sleep(0.05)
    with log.open("a") as fh:
        fh.write(json.dumps({"unit": stem, "sealed_before_next": gens.is_file()}) + "\n")
"""


def test_a_capability_unit_is_sealed_when_written_and_a_retry_generates_only_missing_samples(tmp_path, monkeypatch):
    import os
    import time as _time
    RI = LMD.RI
    from shared import eval_sets as ES
    from stages.calibration_volume import dose_driver
    units = {"stage3_capability_846__u1": [f"a{k:02d}" for k in range(5)],
             "stage3_capability_846__u2": [f"b{k:02d}" for k in range(4)]}
    files = {}
    for k, ids in units.items():
        f = tmp_path / "set" / f"{k.split('__')[1]}.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps({"id": i, "messages": []}) + "\n" for i in ids))
        files[k.split("__")[1]] = f
    monkeypatch.setattr(ES, "manifest", lambda *_a, **_k: {"components": {"stage3_capability_846": {
        "units": {"u1": {}, "u2": {}}}}})
    monkeypatch.setattr(ES, "unit_path", lambda _s, _c, u: files[u])

    def legacy(dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps({"id": f"l{k}", "messages": []}) + "\n" for k in range(3)))
        return dest
    monkeypatch.setattr(dose_driver, "legacy_prompts", legacy)
    real_sleep = _time.sleep
    monkeypatch.setattr(RI.time, "sleep", lambda s: real_sleep(min(s, 0.1)))       #: the runner poll, sped up
    runner = tmp_path / "fake_vllm"
    runner.write_text(FAKE_RUNNER.replace("{python}", sys.executable))
    runner.chmod(0o755)
    log = tmp_path / "runs.jsonl"
    unit_dir = tmp_path / "out" / "eval" / "M-avg_H-T"
    env = {**os.environ, "FAKE_LOG": str(log), "FAKE_FAIL_AFTER": "7"}
    #: attempt 1 fails after 7 samples: u1 (5) is complete and sealed; 2 of u2's are streamed
    with pytest.raises(RI.DriverError, match="exited 3"):
        RI.capability(tmp_path / "pkg", unit_dir, "M-avg_H-T", "H-T", str(runner), env, "att1")
    gens = unit_dir / "generations"
    assert (gens / "stage3_capability_846__u1.jsonl").is_file()
    assert not (gens / "stage3_capability_846__u2.jsonl").is_file()
    runs = [json.loads(l) for l in log.read_text().split("\n") if l.strip()]
    assert {"unit": "stage3_capability_846__u1", "sealed_before_next": True} in runs
    sealed = RI.capability_sealed(unit_dir, "stage3_capability_846__u2", units["stage3_capability_846__u2"], None)
    assert sorted(sealed) == ["b00", "b01"]
    #: attempt 2 asks only for what has no sealed record
    log.unlink()
    env["FAKE_FAIL_AFTER"] = "-1"
    res = RI.capability(tmp_path / "pkg", unit_dir, "M-avg_H-T", "H-T", str(runner), env, "att2")
    assert res["status"] == "complete" and res["sealed_samples_taken"] == {"stage3_capability_846__u2": 2}
    asked = {r["unit"]: r["ids"] for r in (json.loads(l) for l in log.read_text().split("\n") if l.strip())
             if "ids" in r}
    assert asked == {"stage3_capability_846__u2": ["b02", "b03"], "stage3_legacy_150__items": ["l0", "l1", "l2"]}
    u2 = [json.loads(l) for l in (gens / "stage3_capability_846__u2.jsonl").read_text().split("\n") if l.strip()]
    assert [r["id"] for r in u2] == units["stage3_capability_846__u2"]
    assert RI.read_json(gens / "stage3_capability_846__u2.summary.json")["sealed_before"] == 2
    #: a sealed sample whose seed is not the rule's is refused, never taken
    p0 = sorted((unit_dir / "parts" / "stage3_capability_846__u2").glob("part_*.jsonl"))[0]
    bad = [json.loads(l) for l in p0.read_text().split("\n") if l.strip()]
    bad[0]["sampling_seed"] = 1
    p0.write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(RI.DriverError, match="seed"):
        RI.capability_sealed(unit_dir, "stage3_capability_846__u2", units["stage3_capability_846__u2"],
                             lambda i: -1)


def test_the_runner_writes_a_units_generations_before_its_summary():
    """What the sealer relies on, in the real runner: generations, then the summary, each renamed into place."""
    src = (REPO / "scripts/shared/evaluation/uncapped_eval.py").read_text()
    g = src.index('write_atomic(out.with_suffix(".generations.jsonl")')
    assert g < src.index("write_atomic(out, json.dumps(summary") and "--stream-completed" in src
    assert "out.write_text(json.dumps(summary" not in src


# ---- the dev box keeps D plus each candidate's delta --------------------------------------------------------------------

@torch.no_grad()
def test_retain_checks_the_pulled_delta_against_ds_local_copy(teacher, tmp_path):
    from safetensors.torch import save_file
    from reasoning_inheritance import layer_merge_launch as L
    RI = LMD.RI
    sel = {"n_root_blocks": 6, "kept_blocks": [0, 1, 3, 5]}
    spans = [s for s in LMX.LM.spans_from_kept_blocks(6, sel["kept_blocks"]) if len(s) > 1]
    d = FM.keep_blocks(ADAPTER, teacher, sel["kept_blocks"])
    d_dir = tmp_path / "D"
    RI._save(ADAPTER, d, d_dir)
    digests = {n: FM.tensor_sha256(t) for n, t in d.state_dict().items() if n != "lm_head.weight"}
    model, _ = LMX.build_candidate(ADAPTER, teacher, sel, spans, "M-avg", None, digests, seed=0)
    delta = LMX.candidate_delta(sel, spans, model)
    out = tmp_path / "scr" / "store" / "out"
    dp = out / "products" / "M-avg" / "delta.safetensors"
    dp.parent.mkdir(parents=True)
    save_file(delta, str(dp))
    rec = {"identity": {"sha256": "e" * 64}, "status": "durable", "reconstruction": {"equal": True},
           "checkpoint": {"artifact_digest": "f" * 64}, "relay": {"delta_path": "x/delta.safetensors"},
           "delta": {"path": "products/M-avg/delta.safetensors", "sha256": RI.sha256_file(dp),
                     "bytes": dp.stat().st_size, "tensors": len(delta),
                     "tensor_digests": LMX.candidate_tensor_digests(model, sorted(delta))}}
    RI.write_json(out / "records" / "candidates" / "M-avg.json", rec)
    ident = {"weights_sha256": RI.sha256_file(d_dir / "model.safetensors"), "artifact_digest": "d" * 64}
    doc = L.retain("run1", tmp_path / "scr", store=tmp_path / "keep", record=tmp_path / "local.json",
                   d_checkpoint=d_dir, d_identity=ident)
    kept = Path(doc["candidates"]["M-avg"]["path"])
    assert kept.is_file() and kept.stat().st_ino == dp.stat().st_ino     #: one physical copy
    assert doc["base"]["weights_sha256"] == ident["weights_sha256"] and "M-act" not in doc["candidates"]
    #: a delta that is not the candidate's tensors is refused
    rec["delta"]["tensor_digests"] = {n: "0" * 64 for n in rec["delta"]["tensor_digests"]}
    RI.write_json(out / "records" / "candidates" / "M-avg.json", rec)
    with pytest.raises(Exception, match="not the candidate's tensor"):
        L.retain("run1", tmp_path / "scr", store=tmp_path / "keep2", record=tmp_path / "local2.json",
                 d_checkpoint=d_dir, d_identity=ident)
    with pytest.raises(Exception, match="is not D"):
        L.retain("run1", tmp_path / "scr", store=tmp_path / "keep3", record=tmp_path / "local3.json",
                 d_checkpoint=d_dir, d_identity={**ident, "weights_sha256": "0" * 64})


# ---- pre-launch fixes (maintainer review of 34b0fd3a) --------------------------------------------------------------------

def test_a_failed_check_is_never_reused_and_is_run_again(tmp_path):
    """The same identity says what was checked, never that it held: a failed record is kept as evidence and the check
    re-run; only ``passed is True`` is reused."""
    RI = LMD.RI
    out = tmp_path / "out"
    ids = {"checks": LMD.identity("checks", statistics="s")}
    status = RI.Status(tmp_path / "st")
    calls = []

    def run(passed):
        def f():
            calls.append(passed)
            return {"passed": passed, "worst": 1e-6 if passed else 5e-3, "tolerance": 1e-4, "non_finite": []}
        return f
    with pytest.raises(LMD.DriverError, match="changes a block's forward"):
        LMD.passed_checks(out, ids, "att1", status, run(False))
    assert RI.read_json(out / "records" / "merge_checks.json")["passed"] is False      #: the failure is recorded
    #: attempt 2, same identity: the failed record is not accepted -- kept as evidence, the check run again
    assert LMD.passed_checks(out, ids, "att2", status, run(True))["passed"] is True
    moved = out / "superseded" / "att2"
    assert RI.read_json(moved / "records" / "merge_checks.json")["passed"] is False
    assert "did not pass" in RI.read_json(moved / "why.json")[0]["why"]
    #: attempt 3: the passed record is reused, nothing is run
    assert LMD.passed_checks(out, ids, "att3", status, run(False))["passed"] is True and calls == [False, True]
    #: anything but True is not a pass
    RI.write_json(out / "records" / "merge_checks.json", {"passed": "true", "identity": ids["checks"]})
    with pytest.raises(LMD.DriverError):
        LMD.passed_checks(out, ids, "att4", status, run(None))


def test_a_built_not_yet_durable_delta_crosses_to_the_next_attempt(tmp_path):
    """A relay upload or its read-back failed after the build: the dev box's pulled delta goes into the next attempt's
    resume archive and is taken over, so only the relay copy is redone, never the merge."""
    RI = LMD.RI
    from reasoning_inheritance import layer_merge_launch as L
    from stages.reasoning_inheritance import launch as RIL
    scr = tmp_path / "scr"
    out = scr / "store" / "out"
    dp = out / "products" / "M-avg" / "delta.safetensors"
    dp.parent.mkdir(parents=True)
    dp.write_bytes(b"verified delta bytes")
    built = {"identity": {"sha256": "a" * 64}, "status": "built", "reconstruction": {"equal": True},
             "delta": {"path": "products/M-avg/delta.safetensors", "sha256": RI.sha256_file(dp)}}
    RI.write_json(out / "records" / "candidates" / "M-avg.json", built)
    RI.write_json(out / "records" / "candidates" / "M-act.json",
                  {"identity": {"sha256": "b" * 64}, "status": "durable", "relay": {"verified": True},
                   "delta": {"path": "products/M-act/delta.safetensors", "sha256": "0" * 64}})
    extra = L.carried_deltas(scr)
    assert extra == ["products/M-avg/delta.safetensors"]          #: a durable one comes back from the relay instead
    tgz = RIL.resume_archive(scr, tmp_path / "resume.tgz", tuple(extra))
    for subdirs, want in ((("records", "eval"), "absent"), (("records", "eval", "products"), "built")):
        nxt = tmp_path / f"next_{len(subdirs)}" / "out"
        RI.take_over(tgz, nxt, "att2", subdirs=subdirs)
        _ids(nxt, {"M-avg": "a" * 64, "M-act": "b" * 64})
        assert LMD.candidate_state(nxt, "M-avg", "att2") == want   #: without products the build would be redone
        assert LMD.candidate_state(nxt, "M-act", "att2") == "durable"
    #: a pulled copy that does not re-hash is not carried
    dp.write_bytes(b"torn")
    assert L.carried_deltas(scr) == []


@torch.no_grad()
def test_a_restored_delta_lands_where_its_record_says_and_is_retained(teacher, tmp_path):
    """The relay's delta, verified, is renamed into ``out / rec["delta"]["path"]`` (the poll pulls it, ``retain``
    keeps it), never left only in the scratch; a re-pulled copy of a delta already kept becomes the same file."""
    from safetensors.torch import save_file
    from reasoning_inheritance import layer_merge_launch as L
    RI = LMD.RI
    sel = {"n_root_blocks": 6, "kept_blocks": [0, 1, 3, 5]}
    spans = [s for s in LMX.LM.spans_from_kept_blocks(6, sel["kept_blocks"]) if len(s) > 1]
    d = FM.keep_blocks(ADAPTER, teacher, sel["kept_blocks"])
    d_dir = tmp_path / "D"
    RI._save(ADAPTER, d, d_dir)
    digests = {n: FM.tensor_sha256(t) for n, t in d.state_dict().items() if n != "lm_head.weight"}
    model, _ = LMX.build_candidate(ADAPTER, teacher, sel, spans, "M-avg", None, digests, seed=0)
    delta = LMX.candidate_delta(sel, spans, model)
    relay_copy = tmp_path / "relay" / "delta.safetensors"
    relay_copy.parent.mkdir()
    save_file(delta, str(relay_copy))
    rec = {"identity": {"sha256": "e" * 64}, "status": "durable", "reconstruction": {"equal": True},
           "checkpoint": {"artifact_digest": "f" * 64},
           "relay": {"relay": "org/relay", "delta_path": "x/M-avg/eeee/delta.safetensors", "verified": True},
           "delta": {"path": "products/M-avg/delta.safetensors", "sha256": RI.sha256_file(relay_copy),
                     "bytes": relay_copy.stat().st_size, "tensors": len(delta),
                     "tensor_digests": LMX.candidate_tensor_digests(model, sorted(delta))}}
    ident = {"weights_sha256": RI.sha256_file(d_dir / "model.safetensors"), "artifact_digest": "d" * 64}

    def download(r, cache):
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "blob").write_bytes(relay_copy.read_bytes())
        return cache / "blob"
    kept = None
    for run in ("att1", "att2"):                     #: two attempts each restore the durable candidate
        out = tmp_path / run / "scr" / "store" / "out"
        RI.write_json(out / "records" / "candidates" / "M-avg.json", rec)
        dp, src = LMD.fetch_delta(out, tmp_path / run / "work", "M-avg", rec, download)
        assert dp == out / rec["delta"]["path"] and src == "org/relay:x/M-avg/eeee/delta.safetensors"
        assert not (tmp_path / run / "work" / "deltas").exists() and RI.sha256_file(dp) == rec["delta"]["sha256"]
        assert LMD.fetch_delta(out, tmp_path / run / "work", "M-avg", rec, None)[1] == "this attempt's delta file"
        doc = L.retain(run, tmp_path / run / "scr", store=tmp_path / "keep", record=tmp_path / "local.json",
                       d_checkpoint=d_dir, d_identity=ident)
        kept = Path(doc["candidates"]["M-avg"]["path"])
        assert kept.stat().st_ino == dp.stat().st_ino         #: one physical copy, however many attempts pulled it
    #: a relay copy that does not re-hash leaves nothing at the declared path
    out = tmp_path / "att3" / "out"
    relay_copy.write_bytes(b"corrupt")
    with pytest.raises(LMD.DriverError, match="does not hash"):
        LMD.fetch_delta(out, tmp_path / "att3" / "work", "M-avg", rec, download)
    assert not (out / rec["delta"]["path"]).exists() and kept.is_file()

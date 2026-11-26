"""DMR1 variant 1 (the alpha merge at SOB1's depth structure) and the round's shared chain, at $0.

* the design's pins: the spans are SOB1's committed mapping's, the protocol and the registered items are SOB1's, the
  base digests are the base's, the proposals reproduce;
* the alpha merge on the toy teacher with a 4-block span: each merged slot is sum_i alpha_i A_i(theta_i) with alpha the
  importance normalized over the span, recomputed here from the definitions; an all-zero slot is uniform; single
  spans and every other tensor are the base's;
* the alignments are durable pair by pair and reused only under the same correlation;
* the build's products are durable only when the relay verifies them, and are never sent twice;
* the plan is SOB1's depth unit outside the declared fields;
* the registered comparison's family rule (thinking at 0.05, non-thinking only after it rejects; formal only when
  complete) and its per-seed reading;
* the launcher: its kinds and grants, an artifact that does not fund the arm is refused at $0.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stages.depth_merge import common as DC
from stages.depth_merge import merge as M

REPO = Path(__file__).resolve().parents[5]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- the design's pins --------------------------------------------------------------------------------------------

def test_the_spans_are_the_committed_mappings_and_the_protocol_is_sob1s():
    d = DC.design()
    spans = M.merge_spans(d)
    assert [len(s) for s in spans] == [3, 3, 2, 6, 3, 5, 2, 3, 2] and len(M.pairs(spans)) == 20
    assert sum(len(s) for s in M.all_spans(d)) == 36 and len(M.all_spans(d)) == 16
    sob = json.loads((REPO / d["baseline"]["sob1_design"]["path"]).read_text())
    assert d["evaluation"]["protocol"] == sob["evaluation"]["protocol"]
    assert d["analysis"]["primary_items"] == sob["analysis"]["primary_items"]
    assert d["analysis"]["units_of_independence"] == sob["analysis"]["units_of_independence"]
    assert d["teacher"] == sob["teacher"]
    for k in ("sob1_design", "init_record", "training_plan", "study_completion"):
        assert sha(REPO / d["baseline"][k]["path"]) == d["baseline"][k]["sha256"], k
    dig = json.loads(M.DIGESTS.read_text())
    assert dig["checkpoint"]["weights_sha256"] == d["baseline"]["init_record"]["model_safetensors_sha256"]
    assert dig["n_tensors"] == 16 * 11 + 2


def test_the_proposals_and_estimates_reproduce():
    from stages.depth_merge import proposal as P
    for p, doc in P.documents().items():
        assert p.read_text() == P.render(doc), p
    t = json.loads((DC.LOGS / "plans/proposal_dmr1_merge_train.json").read_text())
    b = json.loads((DC.LOGS / "plans/proposal_dmr1_merge_build.json").read_text())
    assert t["status"] == b["status"] == "proposed" and t["authorizes"].startswith("nothing")
    total = t["per_session_hard_cap_usd"] + b["per_session_hard_cap_usd"] + 2 * P.TEARDOWN_RESERVE
    assert total == pytest.approx(t["variant_1_total_with_reserves_usd"])
    assert total <= P.POSITION["project_remaining_usd"]          #: fits the cap's remainder as read 2026-11-26


# ---- the alpha merge on the toy teacher ---------------------------------------------------------------------------

def _batches(n=3, t=40, vocab=128):
    g = torch.Generator().manual_seed(5)
    seqs = [torch.randint(0, vocab, (1, t), generator=g) for _ in range(n)]

    def make():
        for s in seqs:
            yield SimpleNamespace(batch=SimpleNamespace(input_ids=s, attention_mask=torch.ones_like(s)),
                                  original_indices=(0,))
    return make


TOY_SEL = {"n_root_blocks": 6, "kept_blocks": [0, 1, 5], "source": {"seed": 0}}      #: spans [0], [1..4], [5]


def _toy(teacher):
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.planning.layer_merge import spans_from_kept_blocks
    from aadistill.initialization.specs.arch import get_adapter
    adapter = get_adapter("qwen3")
    spans = [s for s in spans_from_kept_blocks(6, TOY_SEL["kept_blocks"]) if len(s) > 1]
    base = FM.keep_blocks(adapter, teacher, TOY_SEL["kept_blocks"])
    digests = {n: FM.tensor_sha256(t) for n, t in base.state_dict().items() if n != "lm_head.weight"}
    return adapter, spans, base, digests


@torch.no_grad()
def test_each_merged_slot_is_the_importance_normalized_sum_of_the_aligned_blocks(teacher, tmp_path):
    LMX = M.lmx()
    adapter, spans, base, digests = _toy(teacher)
    assert spans == [(1, 2, 3, 4)]
    n_q, n_kv, hd = adapter.head_groups(adapter.spec_of(teacher))
    st = LMX.collect_statistics(adapter, teacher, spans, _batches(), None, chunk=16)
    written = {}
    al, report = M.pair_alignments(st, n_q, n_kv, on_pair=lambda tag, x: written.__setitem__(tag, x))
    assert sorted(written) == ["1->2", "1->3", "1->4"] and set(report.values()) == {"computed"}
    stats = LMX.statistics_tensors(st, al)
    model, rec = LMX.build_candidate(adapter, teacher, TOY_SEL, spans, M.METHOD, stats, digests, seed=0)
    #: every tensor outside the merged block is the base's, the merged block is not
    assert rec["checks"]["tensors_outside_spans_differ"] == [] and all(rec["checks"]["span_blocks_differ_from_D"].values())
    #: the gate rows, recomputed from the definitions: fold, align, alpha = I / sum I per neuron
    blocks = adapter.blocks(teacher)
    span = spans[0]
    g_al, s_al = [], []
    for k, b in enumerate(span):
        gate = blocks[b].mlp.gate_proj.weight.double() * blocks[b].post_attention_layernorm.weight.double()[None, :]
        s = stats[f"block{b}.ffn_mean_abs"].double() * blocks[b].mlp.down_proj.weight.double().norm(dim=0)
        if k:
            perm = stats[f"pair{span[0]}_{b}.ffn_perm"]
            gate, s = gate[perm], s[perm]
        g_al.append(gate)
        s_al.append(s)
    alpha = torch.stack(s_al) / torch.stack(s_al).sum(0)
    assert torch.allclose(alpha.sum(0), torch.ones_like(alpha[0]))
    want = sum(alpha[k][:, None] * g_al[k] for k in range(len(span)))
    got = adapter.blocks(model)[1].mlp.gate_proj.weight.double()
    assert torch.allclose(got, want.float().double(), atol=1e-6)
    merged = adapter.blocks(model)[1]
    assert torch.equal(merged.post_attention_layernorm.weight, torch.ones_like(merged.post_attention_layernorm.weight))
    #: the single spans are the base's, bit for bit
    for i in (0, 2):
        for (n, a), (_, b_) in zip(adapter.blocks(model)[i].state_dict().items(), adapter.blocks(base)[i].state_dict().items()):
            assert torch.equal(a, b_), n


@torch.no_grad()
def test_an_all_zero_slot_takes_uniform_weights_and_is_counted(teacher):
    LMX = M.lmx()
    adapter, spans, _base, _dig = _toy(teacher)
    n_q, n_kv, _ = adapter.head_groups(adapter.spec_of(teacher))
    st = LMX.collect_statistics(adapter, teacher, spans, _batches(), None, chunk=16)
    al, _ = M.pair_alignments(st, n_q, n_kv)
    stats = LMX.statistics_tensors(st, al)
    for b in spans[0]:
        stats[f"block{b}.ffn_mean_abs"][3] = 0.0              #: neuron slot 3's importance is zero in every block
        perm = stats.get(f"pair1_{b}.ffn_perm")
        if perm is not None:                                  #: the absorbed block's neuron that lands in slot 3
            stats[f"block{b}.ffn_mean_abs"][int(perm[3])] = 0.0
    _t, how = LMX.merged_tensors(adapter, teacher, spans[0], M.METHOD, stats)
    assert how["uniform_fallback_slots"]["ffn"] >= 1


def test_alignments_are_reused_only_under_the_same_correlation(teacher):
    LMX = M.lmx()
    adapter, spans, _b, _d = _toy(teacher)
    n_q, n_kv, _ = adapter.head_groups(adapter.spec_of(teacher))
    st = LMX.collect_statistics(adapter, teacher, spans, _batches(), None, chunk=16)
    written = {}
    al1, _ = M.pair_alignments(st, n_q, n_kv, on_pair=lambda tag, x: written.__setitem__(tag, x))
    again = {}
    al2, rep = M.pair_alignments(st, n_q, n_kv, persisted=written, on_pair=lambda t, x: again.__setitem__(t, x))
    assert set(rep.values()) == {"reused (same correlation)"} and not again
    for k in al1:
        assert torch.equal(al1[k]["ffn_perm"], al2[k]["ffn_perm"])
    stale = {k: {**v, "correlation_sha256": "0" * 64} for k, v in written.items()}
    _al3, rep3 = M.pair_alignments(st, n_q, n_kv, persisted=stale, on_pair=lambda t, x: again.__setitem__(t, x))
    assert set(rep3.values()) == {"recomputed (another correlation)"} and sorted(again) == sorted(written)


def test_the_base_check_refuses_a_rebuilt_base_that_is_not_the_bases(teacher):
    from stages.depth_merge import build_driver as BD
    adapter, _spans, _base, digests = _toy(teacher)
    BD.check_base(adapter, teacher, TOY_SEL, digests)
    bad = dict(digests)
    bad["model.norm.weight"] = "0" * 64
    with pytest.raises(BD.DriverError):
        BD.check_base(adapter, teacher, TOY_SEL, bad)


# ---- the build's durability ---------------------------------------------------------------------------------------

def _build_record(out: Path, work: Path) -> dict:
    from stages.depth_merge import build_driver as BD
    (work / "products/init").mkdir(parents=True)
    (work / "products/init/model.safetensors").write_bytes(b"weights")
    (work / "products/init/config.json").write_text("{}")
    rec_dir = BD.records(out)
    rec_dir.mkdir(parents=True)
    (rec_dir / f"{M.STATS}.safetensors").write_bytes(b"stats")
    rec = {"run_id": "r", "root": {"root_teacher_id": "t", "root_teacher_sha256": "s"}, "method": M.METHOD,
           "method_identity": {}, "statistics": {"tensors_sha256": sha(rec_dir / f"{M.STATS}.safetensors")},
           "checks": {}, "candidate": {}, "design_sha256": "d",
           "checkpoint": {"artifact_digest": "a" * 64, "shards": [{"filename": "model.safetensors",
                                                                     "sha256": sha(work / "products/init/model.safetensors"),
                                                                     "size_bytes": 7}],
                          "files_sha256": {"model.safetensors": sha(work / "products/init/model.safetensors"),
                                           "config.json": sha(work / "products/init/config.json")}}}
    (rec_dir / "build.json").write_text(json.dumps(rec))
    return rec


def test_the_init_record_is_complete_only_when_every_relay_copy_verifies_and_nothing_is_sent_twice(tmp_path):
    from stages.depth_merge import build_driver as BD
    from stages.reasoning_inheritance.driver import Status
    out, work = tmp_path / "out", tmp_path / "work"
    _build_record(out, work)
    sent = []

    def put(payload, timeout_s, ok=True):
        sent.append(payload["prefix"])
        files = {k: sha(v) for k, v in payload["files"].items()}
        return {"relay": "r", "prefix": payload["prefix"], "files_sha256": files if ok else {}, "verified": ok}
    with pytest.raises(BD.DriverError):
        BD.persist(out, work, "r", Status(tmp_path / "st"), put=lambda p, t: put(p, t, ok=False))
    assert BD.durable_init(out) is None and not (out / BD.ARM / "init_record.json").exists()
    sent.clear()
    doc = BD.persist(out, work, "r", Status(tmp_path / "st"), put=put)
    assert doc["status"] == "COMPLETE" and BD.product_problems(doc) == [] and BD.durable_init(out)
    assert [s.split("/")[-2] for s in sent] == ["init", "statistics"]
    sent.clear()
    BD.persist(out, work, "r", Status(tmp_path / "st"), put=put)            #: a takeover: verified copies stay
    assert sent == []
    from stages.depth_merge import launch as L
    from stages.depth_merge import session as S
    scr = tmp_path / "scr"
    (scr / "store").mkdir(parents=True)
    (scr / "store/out").symlink_to(out)
    res = S.completion(L.VARIANT, L.BUILD_KIND, None, "r1", scr, root=tmp_path)
    assert res["status"] == "COMPLETE"


# ---- the comparison's rule ----------------------------------------------------------------------------------------

def _y(acc: dict, items, seeds):
    """``{mode: {seed: {item: 0/1}}}`` with the first round(acc * n) items correct."""
    out = {}
    for mode, a in acc.items():
        k = round(a * len(items))
        out[mode] = {s: {i: int(j < k) for j, i in enumerate(items)} for s in seeds}
    return out


def test_each_comparison_is_its_own_fixed_sequence_family_and_formal_only_when_complete():
    from stages.depth_merge import analysis as A
    items = [f"i{j}" for j in range(80)]
    groups = {f"g{j}": [items[2 * j], items[2 * j + 1]] for j in range(40)}
    seeds = [1, 2]
    better = _y({"H-T": 0.6, "H-N": 0.6}, items, seeds)
    worse = _y({"H-T": 0.2, "H-N": 0.2}, items, seeds)
    res = A.compare(better, worse, groups, seeds, ("a", "b"))
    ht, hn = res["modes"]["H-T"], res["modes"]["H-N"]
    assert ht["status"] == hn["status"] == "formal" and ht["verdict"] == "a_BETTER" and hn["verdict"] == "a_BETTER"
    assert ht["seeds_agree_in_sign"] and set(ht["per_seed"]) == {"1", "2"}
    #: the thinking test does not reject: non-thinking is descriptive, never tested
    same = A.compare(worse, worse, groups, seeds, ("a", "b"))
    assert same["modes"]["H-T"]["verdict"] == "NOT_DISTINGUISHED" and same["modes"]["H-N"]["status"] == "descriptive_only"
    assert "p_two_sided" not in same["modes"]["H-N"]
    #: a seed missing: descriptive in every mode
    part = copy.deepcopy(better)
    del part["H-T"][2]
    miss = A.compare(part, worse, groups, seeds, ("a", "b"))
    assert miss["modes"]["H-T"]["status"] == miss["modes"]["H-N"]["status"] == "descriptive_only"


def test_the_comparison_registry_names_the_three_direct_questions():
    from stages.depth_merge import analysis as A
    c = A.comparisons()
    assert c["C1"]["pair"] == ["depth_merge", "depth_only"] and c["C3"]["pair"] == ["depth_loop_hidden", "depth_loop"]
    assert A.result_path("C1").parent == DC.LOGS / "results"


# ---- the plan -----------------------------------------------------------------------------------------------------

def merge_init_record(root: Path) -> Path:
    """A COMPLETE variant-1 init record as the build writes it, at a committed-like path under ``root``."""
    base = json.loads((REPO / DC.design()["baseline"]["init_record"]["path"]).read_text())
    w = "b" * 64
    doc = {"schema": "aadistill.stage1.depth_merge.init_record/v1", "status": "COMPLETE", "arm": "depth_merge",
           "root_teacher": base["root_teacher"],
           "commit": {"selected_rows": [{"state_id": "depth_merge/x", "artifact_digest": "c" * 64,
                                         "single_shard_sha256": w}]},
           "checkpoint": {"artifact_digest": "c" * 64, "shards": [{"filename": "model.safetensors", "sha256": w,
                                                                     "size_bytes": 4007724000}],
                          "files_sha256": {"model.safetensors": w,
                                           "config.json": DC.design()["baseline"]["init_record"]["config_json_sha256"]},
                          "num_parameters_counted": 2003851776, "num_parameters": 2003851776},
           "relay": {"init": {"relay": "AlphaAvatar/aadistill-artifacts", "prefix": "durable/stage1_depth_merge/depth_merge/init/cccc",
                              "files_sha256": {"init/model.safetensors": w, "init/config.json": DC.design()["baseline"]["init_record"]["config_json_sha256"]},
                              "bytes": {"init/model.safetensors": 4007724000, "init/config.json": 1154}, "verified": True},
                     "statistics": {"verified": True}}}
    p = root / "logs/stages/stage-1/depth_merge/runs/dmr1_merge_build_x/evidence/depth_merge/init_record.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    return p


def link_into(root: Path, rel: str) -> None:
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.symlink_to(REPO / rel)


def test_the_merge_arms_plan_is_sob1s_depth_unit_outside_the_declared_fields(tmp_path):
    from stages.depth_merge import plan as PL
    link_into(tmp_path, DC.design()["baseline"]["training_plan"]["path"])
    doc = PL.derive("depth_merge", merge_init_record(tmp_path), root=tmp_path)
    assert [u["unit"] for u in doc["units"]] == ["depth_merge-s20260726", "depth_merge-s20260801"]
    for u in doc["units"]:
        c = u["config"]
        assert c["stage"] == "stage3_recovery" and "hidden_state" not in c["loss"]
        assert c["lineage"]["initialization"]["model_safetensors_sha256"] == "b" * 64
        assert set(doc["derivation_checks"][u["unit"]]["differs_from_sob1_depth_unit"]) <= {
            "run_name", "student_path", "out_dir", "lineage.initialization.model_safetensors_sha256",
            "lineage.initialization.provenance.record", "lineage.initialization.provenance.record_sha256"}
        assert c["schedule"]["total_steps"] == 9701 and c["block_len"] == 8192
    assert doc["relay_prefix_root"] == "durable/stage1_depth_merge"
    bad = json.loads(merge_init_record(tmp_path).read_text())
    bad["relay"]["init"]["verified"] = False
    p = tmp_path / "logs/bad_init.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(PL.PlanError):
        PL.derive("depth_merge", p, root=tmp_path)


# ---- the launcher -------------------------------------------------------------------------------------------------

def test_the_launchers_kinds_grants_and_an_artifact_for_another_arm_is_refused(tmp_path, monkeypatch, capsys):
    from stages.depth_merge import launch as L
    from stages.depth_merge import session as S
    from stages.depth_recurrence import launch as L6
    assert L.STUDY.kinds == ("dmr1_merge_build", "dmr1_merge_train") and L.STUDY.formal
    assert L6.STUDY.kinds == ("dmr1_loop_train", "dmr1_loop_hidden_train")
    assert {k: str(g) for k, g, _ in L.STUDY.kind_grants} == {
        "dmr1_merge_build": "logs/budget/approvals/dmr1_dmr1_merge_build_grant.json",
        "dmr1_merge_train": "logs/budget/approvals/dmr1_dmr1_merge_train_grant.json"}
    cond = L.kind_conditions(L.BUILD_KIND)
    assert cond["digests_sha256"] == sha(M.DIGESTS) and "identity:configs/stages/stage-1/depth_merge/design.json" in cond
    #: nothing is granted: a real issue refuses at $0
    for k in L.STUDY.kinds:
        assert not (REPO / L.VARIANT.grant_path(k)).exists()
    with pytest.raises(S.SE.AuthorizationError):
        S.kind_grant(L.VARIANT, L.TRAIN_KIND)
    #: the loop kinds have no proposal this round: budget refuses
    with pytest.raises(S.SE.AuthorizationError):
        S.kind_grant(L6.VARIANT, L6.LOOP_KIND, dry_run=True)
    with pytest.raises(S.SE.AuthorizationError, match="no grant and no proposal"):
        L6.launch_spec(L6.LOOP_KIND)
    assert L.launch_spec(L.BUILD_KIND).driver.endswith("build_driver.py")
    #: an artifact naming another arm's plan is refused before anything is read
    sc = {"kind": L.TRAIN_KIND, "arm": "depth_merge", "plan": "x", "plan_sha256": "y", "init_record_sha256": "z"}
    monkeypatch.setattr(S, "kind_conditions", lambda v, kind, root=REPO, arm=None: {**sc, "plan_sha256": "other"})
    assert S.arm_binding_problems(L.VARIANT, sc, "depth_merge")

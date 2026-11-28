"""DMR1 v1-new (arm ``depth_merge_split``), at $0.

* the groups derive from SOB1's committed selection and the threshold alone: 18 layers, the two long runs split off,
  every absorbed group one of V1's spans; the config is the base's file with 18 layers and its parameter count, on
  the meta device, is 2,205,713,408; the design is V1's outside its declared keys and is the derivation;
* the candidate on the toy teacher: a kept group is the base's layer, an absorbed group V1's built layer byte for byte,
  a separate group the alpha merge over the run's own blocks only (alpha normalized within the run, the run's first
  block the anchor) -- never V1's span block with the kept block in front;
* the build's identity is its own (the groups, the reuse of V1's init) and binding it leaves V1's untouched; the plan is
  SOB1's depth unit of one seed at v1-new's sizes; the launcher's v1-new kinds bind v1-new's files and V1's kinds do
  not; the programme's storage arithmetic is each session's own launch-gate need.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from stages.depth_merge import common as DC
from stages.depth_merge import merge as M
from stages.depth_merge import split as SPLIT

REPO = Path(__file__).resolve().parents[5]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _v1_tests():
    """variant 1's test module (its toy batches, its init-record fixture), loaded by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dmr1_v1_tests", Path(__file__).with_name("test_depth_merge.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_groups_config_parameters_and_design_derive_from_the_selection_and_the_threshold():
    g = SPLIT.groups()
    assert len(g) == 18 and SPLIT.groups_problems() == []
    assert SPLIT.new_spans(g) == [(13, 14, 15, 16, 17), (25, 26, 27, 28)]
    assert [x["anchor"] for x in g if x["kind"] == "kept"][:9] == [0, 4, 5, 6, 12, 18, 22, 23, 24]
    assert set(SPLIT.absorbed_spans(g)) <= {tuple(s) for s in M.merge_spans(DC.design())}
    #: the kept block before each separate run is left as it is (1-based 13 and 25)
    assert [g[i - 1]["layers"] for i, x in enumerate(g) if x["kind"] == "separate"] == [(12,), (24,)]
    assert SPLIT.CONFIG.read_text() == SPLIT.config_text()
    cfg = json.loads(SPLIT.CONFIG.read_text())
    assert cfg["num_hidden_layers"] == 18 and len(cfg["layer_types"]) == 18
    base = json.loads(SPLIT.BASE_CONFIG.read_text())
    assert {k: v for k, v in cfg.items() if k not in ("num_hidden_layers", "layer_types")} == {
        k: v for k, v in base.items() if k not in ("num_hidden_layers", "layer_types")}
    assert SPLIT.parameter_count() == SPLIT.spec()["num_parameters"] == 2205713408
    sizes = SPLIT.checkpoint_bytes()
    assert sizes["init_bf16"] == 2 * 2205713408 and sizes["final_fp32"] == 4 * 2205713408
    #: the design: V1's outside the declared keys, and exactly what split.py derives
    v1, d = DC.design(), json.loads(SPLIT.DESIGN.read_text())
    assert SPLIT.DESIGN.read_text() == SPLIT.render(SPLIT.design_doc())
    assert {k for k in set(v1) | set(d) if v1.get(k) != d.get(k)} <= set(SPLIT.CHANGED)
    assert d["variant"]["derived_from"]["sha256"] == sha(DC.DESIGN)
    assert d["arms"]["depth_merge_split"]["seeds"] == d["training"]["seeds"] == [20260726]
    assert d["statistics"]["pairs"] == [[13, 14], [13, 15], [13, 16], [13, 17], [25, 26], [25, 27], [25, 28]]
    assert d["analysis"]["test"].startswith("none") and "NOT parameter-matched" in d["analysis"]["structures"]


@torch.no_grad()
def test_a_separate_run_is_merged_from_its_own_blocks_and_an_absorbed_span_is_v1s_layer(teacher, tmp_path):
    from safetensors.torch import save_file
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.planning.layer_merge import groups_from_kept_blocks
    from aadistill.initialization.specs.arch import get_adapter
    _batches = _v1_tests()._batches
    LMX = M.lmx()
    adapter = get_adapter("qwen3")
    kept = [0, 2, 5]
    g = groups_from_kept_blocks(6, kept, separate_run_min=2)
    assert [(x["layers"], x["kind"]) for x in g] == [((0, 1), "absorbed"), ((2,), "kept"), ((3, 4), "separate"),
                                                    ((5,), "kept")]
    n_q, n_kv, _ = adapter.head_groups(adapter.spec_of(teacher))
    base = FM.keep_blocks(adapter, teacher, kept)
    digests = {n: FM.tensor_sha256(t) for n, t in base.state_dict().items() if n != "lm_head.weight"}
    #: V1's init at the toy: its spans (0,1) and (2,3,4), merged as V1 merges them, saved as its file
    v1_spans = [(0, 1), (2, 3, 4)]
    st1 = LMX.collect_statistics(adapter, teacher, v1_spans, _batches(), None, chunk=16)
    al1, _ = M.pair_alignments(st1, n_q, n_kv)
    v1, _ = LMX.build_candidate(adapter, teacher, {"n_root_blocks": 6, "kept_blocks": kept, "source": {"seed": 0}},
                                v1_spans, M.METHOD, LMX.statistics_tensors(st1, al1), digests, seed=0)
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    save_file({n: t.contiguous() for n, t in v1.state_dict().items() if n != "lm_head.weight"},
              str(v1_dir / "model.safetensors"))
    #: v1-new's statistics: the separate run only
    st = LMX.collect_statistics(adapter, teacher, SPLIT.new_spans(g), _batches(), None, chunk=16)
    al, _ = M.pair_alignments(st, n_q, n_kv)
    stats = LMX.statistics_tensors(st, al)
    assert sorted(k for k in stats if k.startswith("pair")) and all(k.startswith(("block3", "block4", "pair3_4"))
                                                                    for k in stats)
    model, rec = SPLIT.build_candidate(adapter, teacher, stats, digests, v1_dir, seed=0, g=g, base_kept=kept)
    c = rec["checks"]
    assert (c["n_kept"], c["n_absorbed"], c["n_separate"], c["num_hidden_layers"]) == (2, 1, 1, 4)
    blocks, b1, bb = adapter.blocks(model), adapter.blocks(v1), adapter.blocks(base)
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))  # noqa: E731
    assert same(blocks[0], b1[0])                         #: absorbed: V1's built layer, byte for byte
    assert same(blocks[1], bb[1]) and same(blocks[3], bb[2])          #: kept: the base's (teacher 2 and 5)
    assert not same(blocks[2], b1[1])                     #: never V1's (2,3,4) block
    #: the separate layer's gate rows: alpha over the run {3, 4} only, anchored at 3
    tb = adapter.blocks(teacher)
    g_al, s_al = [], []
    for k, b in enumerate((3, 4)):
        gate = tb[b].mlp.gate_proj.weight.double() * tb[b].post_attention_layernorm.weight.double()[None, :]
        s = stats[f"block{b}.ffn_mean_abs"].double() * tb[b].mlp.down_proj.weight.double().norm(dim=0)
        if k:
            perm = stats["pair3_4.ffn_perm"]
            gate, s = gate[perm], s[perm]
        g_al.append(gate)
        s_al.append(s)
    alpha = torch.stack(s_al) / torch.stack(s_al).sum(0)
    want = sum(alpha[k][:, None] * g_al[k] for k in range(2))
    assert torch.allclose(blocks[2].mlp.gate_proj.weight.double(), want.float().double(), atol=1e-6)
    assert c["num_parameters"] == sum(p.numel() for p in model.parameters())
    #: a rebuilt base that is not the base's refuses (V1's file itself is checked by its sha256 when fetched)
    with pytest.raises(SPLIT.SplitError):
        SPLIT.build_candidate(adapter, teacher, stats, {**digests, "model.norm.weight": "0" * 64}, v1_dir,
                              seed=0, g=g, base_kept=kept)


def test_the_builds_identity_is_its_own_and_binding_it_leaves_v1s_untouched(tmp_path):
    from stages.depth_merge import build_driver as BD
    v1 = BD.current_identity(tmp_path / "a")
    with BD.bound("depth_merge_split"):
        assert BD.ARM == "depth_merge_split" and BD.SCHEMA.startswith("aadistill.stage1.depth_merge_split")
        sp = BD.current_identity(tmp_path / "b")
        assert BD.group_root("init") == "durable/stage1_depth_merge/depth_merge_split/init"
        ref = BD.v1_init_ref()
    assert BD.ARM == "depth_merge" and v1 == BD.current_identity(tmp_path / "c")
    assert sp["statistics"] != v1["statistics"] and sp["checkpoint"] != v1["checkpoint"]
    assert sp["parts"]["mapping"]["pairs"] == [[13, 14], [13, 15], [13, 16], [13, 17], [25, 26], [25, 27], [25, 28]]
    assert len(sp["parts"]["mapping"]["groups"]) == 18 and "groups" not in v1["parts"]["mapping"]
    assert sp["parts"]["reuse"]["v1_init"]["model_safetensors_sha256"] == ref["model_safetensors_sha256"]
    plan = json.loads((DC.LOGS / "plans/training_plan_depth_merge.json").read_text())
    assert ref["init_record"] == plan["init_record"]
    #: V1's committed init record still stands under the rebound driver
    rec = json.loads((REPO / plan["init_record"]["path"]).read_text())
    assert BD.product_problems(rec, v1) == []


def test_the_plan_is_sob1s_depth_unit_of_one_seed_at_v1news_sizes(tmp_path):
    import tempfile
    from stages.depth_merge import build_driver as BD
    from stages.depth_merge import plan as PL
    V1T = _v1_tests()
    link_into, merge_init_record = V1T.link_into, V1T.merge_init_record
    link_into(tmp_path, DC.design()["baseline"]["training_plan"]["path"])
    p = merge_init_record(tmp_path)
    doc = json.loads(p.read_text())
    with BD.bound("depth_merge_split"), tempfile.TemporaryDirectory() as tmp:
        ident = BD.current_identity(Path(tmp))
        for g in ("statistics", "init"):
            doc["relay"][g]["identity_sha256"] = ident["statistics" if g == "statistics" else "checkpoint"]
        doc["relay"]["init"]["prefix"] = "durable/stage1_depth_merge/depth_merge_split/init/cccc"
        doc["statistics"]["identity"] = BD.stamp(ident, "statistics")
        doc["checkpoint"]["identity"] = BD.stamp(ident, "checkpoint")
        doc = {**doc, "arm": "depth_merge_split", "schema": BD.SCHEMA, "identity": ident,
               "method": BD._checkpoint_method(doc["checkpoint"])}
        assert BD.product_problems(doc, ident) == []
    q = tmp_path / "logs/stages/stage-1/depth_merge/runs/dmr1_split_build_x/evidence/depth_merge_split/init_record.json"
    q.parent.mkdir(parents=True)
    q.write_text(json.dumps(doc))
    plan = PL.derive("depth_merge_split", q, root=tmp_path)
    assert [u["unit"] for u in plan["units"]] == ["depth_merge_split-s20260726"]
    u = plan["units"][0]
    assert u["bytes"] == {"save_point": 23356911616, "final": 8822853632, "init": 4411426816}
    assert u["config"]["schedule"]["total_steps"] == 9701 and u["config"]["stage"] == "stage3_recovery"
    assert plan["design_sha256"] == sha(DC.SPLIT_DESIGN)
    assert plan["seed_policy"] == json.loads(DC.SPLIT_DESIGN.read_text())["training"]["single_seed"]
    #: V1's init record is not v1-new's
    with pytest.raises(PL.PlanError):
        PL.derive("depth_merge_split", p, root=tmp_path)


def test_v1news_kinds_bind_its_files_and_v1s_kinds_do_not(monkeypatch):
    from stages.depth_merge import launch as L
    from stages.depth_merge import session as S
    assert L.STUDY.kinds == ("dmr1_merge_build", "dmr1_split_build", "dmr1_merge_train", "dmr1_split_train")
    assert L.VARIANT.build_kinds == {"dmr1_merge_build": "depth_merge", "dmr1_split_build": "depth_merge_split"}
    v1 = L.kind_conditions(L.BUILD_KIND)
    new = L.kind_conditions(L.SPLIT_BUILD_KIND)
    assert not any("split" in k for k in v1) and "build_arm" not in v1
    assert new["build_arm"] == "depth_merge_split" and new["split_config_sha256"] == sha(SPLIT.CONFIG)
    assert new["identity:configs/stages/stage-1/depth_merge/split_design.json"] == sha(SPLIT.DESIGN)
    #: no v1-new grant exists: it is proposed, never granted here
    for k in (L.SPLIT_BUILD_KIND, L.SPLIT_TRAIN_KIND):
        assert not (REPO / L.VARIANT.grant_path(k)).exists()
        with pytest.raises(S.SE.AuthorizationError):
            S.kind_grant(L.VARIANT, k)
        prop = json.loads((REPO / L.VARIANT.proposal_path(k)).read_text())
        assert prop["status"] == "proposed" and prop["authorizes"].startswith("nothing")
        assert prop["design_sha256"] == sha(SPLIT.DESIGN)
    assert L.VARIANT.arm_of(L.SPLIT_TRAIN_KIND, None) == "depth_merge_split"


def test_the_programme_storage_is_each_sessions_own_gate_need():
    from stages.depth_merge import storage as ST
    GIB = 2 ** 30
    b = ST.session_need("v1new_build")
    assert b["need_bytes"] == ST.sequence(arm="depth_merge_split")["relay"]["need_bytes"]
    #: a programme subtracts what each session leaves and adds what a scenario frees, nothing else
    h = 100 * GIB
    r = ST.program(["v1new_build", "v1new_train"], h)
    assert r["sessions"][1]["headroom_gib"] == round((h - b["leaves_bytes"]) / GIB, 2)
    f = ST.program(["v1new_build", "v1new_train"], h, frees={"v1new_build": ["v1_init"]})
    assert f["sessions"][1]["headroom_gib"] == round((h - b["leaves_bytes"] + ST.FREEABLE["v1_init"][0]) / GIB, 2)
    doc = json.loads((DC.LOGS / "plans/programme_storage_plan.json").read_text())
    again = ST.split_plan(doc["live"])
    assert again == doc

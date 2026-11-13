"""The frozen mapping's builders write what the operators write, and a saved mapping is its bytes.

The builders repeat four operators' surgery on GIVEN selections (`planning/frozen_mapping.py`). If one drifted from
its operator, a mapping taken from a search journal would materialize a different model than the search measured and
nothing downstream could tell. So each builder is run beside its operator on the same toy parent and the children are
compared bit for bit; the width transform is recomputed from the statistics the width operator itself collected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.initialization  # noqa: F401,E402
from aadistill.initialization.operators.base import OperatorContext, get_implementation  # noqa: E402
from aadistill.initialization.operators.attention.gqa.activation_importance import (  # noqa: E402
    ATTENTION_ACTIVATION_IMPORTANCE_V1,
)
from aadistill.initialization.specs.arch import get_adapter  # noqa: E402
from aadistill.initialization.planning import frozen_mapping as FM  # noqa: E402

ADAPTER = get_adapter("qwen3")


def _ctx(model, target_spec, items, profile, cache=None):
    return OperatorContext(adapter=ADAPTER, model=model, parent_spec=ADAPTER.spec_of(model),
                           target_spec=target_spec, profile=profile, calibration_items=items, seed=1234,
                           config={}, stats_cache=cache, stats_cache_key=None if cache is None else "parent")


def _same(a, b) -> list[str]:
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    return [n for n in pa if not torch.equal(pa[n], pb[n])]


@pytest.fixture
def path(teacher, target_spec, calibration_items, profile):
    """The operators' own path FFN -> DEPTH -> WIDTH -> ATTENTION on the toy teacher, with what each selected."""
    ffn = get_implementation("ffn.activation_importance_v0").execute(
        _ctx(teacher, target_spec, calibration_items, profile))
    depth = get_implementation("depth.positional_v0").execute(_ctx(ffn.model, target_spec, calibration_items, profile))
    cache = FM.RetainingStatsCache()
    width = get_implementation("width.global_pca_v0").execute(
        _ctx(depth.model, target_spec, calibration_items, profile, cache))
    attn = ATTENTION_ACTIVATION_IMPORTANCE_V1.execute(_ctx(width.model, target_spec, calibration_items, profile))
    return {"ffn": ffn, "depth": depth, "width": width, "attn": attn, "stats": cache.retained["parent"]}


def _transform(path, depth_model):
    n = ADAPTER.spec_of(depth_model)["num_hidden_layers"]
    weights = path["width"].artifacts["projection_diagnostics"]["point_weights"]
    return FM.width_transform_from_statistics(
        path["stats"], d_child=ADAPTER.spec_of(path["width"].model)["hidden_size"], points=list(range(n + 1)),
        point_weights=weights, final_norm_weight=ADAPTER.final_norm(depth_model).weight, post_norm_point=n)


def test_each_builder_writes_its_operators_child(path):
    kept_layers = path["depth"].trace["kept_layers"]
    kept_heads = path["attn"].artifacts["kept_heads"]
    parent = path["ffn"].model  # whatever the FFN operator's own parent was, its child is the next parent
    assert not _same(FM.keep_blocks(ADAPTER, parent, kept_layers), path["depth"].model)
    t = _transform(path, path["depth"].model)
    assert not _same(FM.project_residual_width(ADAPTER, path["depth"].model, t), path["width"].model)
    assert not _same(FM.select_query_heads(ADAPTER, path["width"].model, kept_heads), path["attn"].model)


def test_the_ffn_builder_writes_the_ffn_operators_child(teacher, target_spec, calibration_items, profile, path):
    built = FM.select_ffn_neurons(ADAPTER, teacher, path["ffn"].artifacts["kept_neurons"])
    assert not _same(built, path["ffn"].model)


def _mapping(path, teacher):
    kept = tuple(path["depth"].trace["kept_layers"])
    n = ADAPTER.spec_of(teacher)["num_hidden_layers"]
    heads = {kept[i]: tuple(h) for i, h in enumerate(path["attn"].artifacts["kept_heads"])}
    return FM.FrozenMapping(
        root={"id": "toy"}, n_root_blocks=n, kept_blocks=kept,
        ffn_neurons={b: tuple(v) for b, v in enumerate(path["ffn"].artifacts["kept_neurons"])},
        query_heads={**{b: (0, 2) for b in range(n) if b not in kept}, **heads},
        width=_transform(path, path["depth"].model))


def test_materialize_reproduces_the_path_and_shares_it_across_depths(teacher, path):
    m = _mapping(path, teacher)
    assert not FM.problems(m, root_spec=ADAPTER.spec_of(teacher))
    dw, roots = FM.materialize(ADAPTER, teacher, m, blocks="kept", non_depth=True)
    assert roots == list(m.kept_blocks) and not _same(dw, path["attn"].model)
    w, all_roots = FM.materialize(ADAPTER, teacher, m, blocks="all", non_depth=True)
    pairs = [(all_roots.index(r), i) for i, r in enumerate(roots)]
    assert not FM.block_differences(ADAPTER, w, dw, pairs)
    assert not FM.shared_parameter_differences(ADAPTER, w, dw)
    d, _ = FM.materialize(ADAPTER, teacher, m, blocks="kept", non_depth=False)
    assert not FM.block_differences(ADAPTER, teacher, d, [(r, i) for i, r in enumerate(roots)])
    assert not FM.shared_parameter_differences(ADAPTER, teacher, d)
    with pytest.raises(FM.FrozenMappingError):
        FM.materialize(ADAPTER, teacher, m, blocks="all", non_depth=False)


def test_a_saved_mapping_is_its_bytes(teacher, path, tmp_path):
    m = _mapping(path, teacher)
    rec = FM.save(m, tmp_path / "m")
    again = FM.load(tmp_path / "m", expected_content_digest=rec["content_digest"])
    assert again.content_digest == m.content_digest
    assert torch.equal(again.width.projection, m.width.projection.cpu())
    (tmp_path / "m" / FM.TENSOR_FILE).write_bytes((tmp_path / "m" / FM.TENSOR_FILE).read_bytes()[:-8] + b"\0" * 8)
    with pytest.raises(FM.FrozenMappingError):
        FM.load(tmp_path / "m")


def test_problems_name_a_mapping_that_does_not_fit(teacher, path):
    m = _mapping(path, teacher)
    bad = FM.FrozenMapping(root=m.root, n_root_blocks=m.n_root_blocks, kept_blocks=(3, 1),
                           query_heads={0: (0, 1)}, provenance={})
    found = " ".join(FM.problems(bad, root_spec=ADAPTER.spec_of(teacher)))
    assert "kept_blocks" in found and "per KV group" in found

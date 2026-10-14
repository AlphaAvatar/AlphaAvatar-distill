"""D2's teacher-confidence validity checks, on a toy root store at $0."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from support.toy import TEACHER_GEOMETRY, build_tiny_model, make_items  # noqa: E402

from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    SUPERVISED_TARGET_V1,
)
from aadistill.initialization.scoring.reference_store import (  # noqa: E402
    build_root_reference_store, root_reference_identity,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    reference_topk_tail,
)
from stages.d_series import teacher_confidence as TC  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    raw = make_items(n_per_subtype=2, seq_len=24)
    for item in raw:
        n = int(item["input_ids"].shape[1]) - 1
        t = torch.arange(n)
        item["tags"] = {"assistant": t >= 4, "reasoning": (t >= 4) & (t < 15),
                        "final_answer": t >= 15, "eos": t == n - 1,
                        "think_close": t == 14}
    items = prepare_calibration_items(raw, profile_id="toy@v1")
    model = build_tiny_model(TEACHER_GEOMETRY)
    support = reference_topk_tail(8)
    store = build_root_reference_store(
        model, items, support=support,
        identity=root_reference_identity(root_identity={"teacher_id": "toy"},
                                         profile_hash="p", support=support),
        micro_batch_size=1, packing="original_order_v1", device="cpu")
    return store, items


def test_the_distribution_is_over_scored_positions_only(setup):
    store, items = setup
    rep = TC.validity_report(store, items, policy=SUPERVISED_TARGET_V1)
    scored = sum(int(SUPERVISED_TARGET_V1.weights(i, axis="prediction")
                     .weights.gt(0).sum()) for i in items)
    d = rep["distribution"]["confidence"]
    assert d["positions"] == scored
    c = torch.cat([store.per_position(i, "confidence").double()[
        SUPERVISED_TARGET_V1.weights(i, axis="prediction").weights > 0]
        for i in items])
    assert d["weight_sum"] == pytest.approx(float(c.sum()))
    assert d["effective_sample_size"] == pytest.approx(
        float(c.sum() ** 2 / (c * c).sum()))
    assert sum(rep["distribution"]["histogram_20_bins_0_1"]) == scored
    assert rep["log_vocab_nats"] == pytest.approx(
        math.log(TEACHER_GEOMETRY["vocab_size"]))


def test_thinking_answer_and_structure_are_separated(setup):
    store, items = setup
    g = TC.validity_report(store, items, policy=SUPERVISED_TARGET_V1)["groups"]
    n_reason = g["span:reasoning"]["positions"]
    n_answer = g["span:final_answer"]["positions"]
    assert n_reason == 4 * 11 and n_answer == 4 * 8
    assert g["structural:eos"]["positions"] == 4
    shares = sum(g[f"span:{t}"].get("weight_share", 0.0)
                 for t in ("reasoning", "final_answer", "other_scored"))
    assert shares == pytest.approx(1.0)
    assert {"domain:general", "domain:math", "subtype:text",
            "subtype:arith"} <= set(g)


def test_token_types_need_a_decoder_and_cover_every_position(setup):
    store, items = setup
    rep = TC.validity_report(store, items, policy=SUPERVISED_TARGET_V1,
                             decode=lambda t: str(t) if t % 3 else " w")
    g = rep["groups"]
    token = {k: v["positions"] for k, v in g.items() if k.startswith("token:")}
    assert sum(token.values()) == rep["distribution"]["confidence"]["positions"]
    assert token["token:structural"] == 4 + 4
    assert not any(k.startswith("token:") for k in TC.validity_report(
        store, items, policy=SUPERVISED_TARGET_V1)["groups"])


def test_the_kl_relation_and_shares(setup):
    store, items = setup
    kl = {str(i["item_id"]): torch.rand(
        int(i["input_ids"].shape[1]) - 1,
        generator=torch.Generator().manual_seed(k))
        for k, i in enumerate(items)}
    rep = TC.validity_report(store, items, policy=SUPERVISED_TARGET_V1, kl=kl)
    r = rep["relation_to_kl"]
    c = torch.cat([store.per_position(i, "confidence").double()[
        SUPERVISED_TARGET_V1.weights(i, axis="prediction").weights > 0]
        for i in items])
    k = torch.cat([kl[str(i["item_id"])].double()[
        SUPERVISED_TARGET_V1.weights(i, axis="prediction").weights > 0]
        for i in items])
    assert r["weighted_over_unweighted_kl"] == pytest.approx(
        float((c * k).sum() / k.sum()))
    shares = r["group_shares"]
    assert sum(shares[f"span:{t}"]["share_of_weighted_objective"]
               for t in ("reasoning", "final_answer", "other_scored")) \
        == pytest.approx(1.0)
    assert any("KL relation" in x for x in rep["risks"])


def test_the_relation_to_the_teachers_own_probability(setup):
    store, items = setup
    rel = TC.validity_report(store, items,
                             policy=SUPERVISED_TARGET_V1)[
        "relation_to_teacher_probability"]
    assert -1.0 <= rel["spearman_confidence_vs_target_log_prob"] <= 1.0
    assert 0.0 <= rel["target_is_top1_fraction"] <= 1.0
    assert sum(d["positions"] for d in
               rel["mean_confidence_by_target_probability_decile"]) > 0


def test_spearman_against_an_independent_case():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert TC.spearman(x, x) == pytest.approx(1.0)
    assert TC.spearman(x, -x) == pytest.approx(-1.0)
    assert TC.spearman(x, torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0])) == \
        pytest.approx(0.9486833, abs=1e-6)
    assert TC.spearman(x[:2], x[:2]) is None


@pytest.mark.parametrize("text,cls", [
    ("12", "digit"), (" 7", "digit"), (",", "punctuation"), (" ", "whitespace"),
    ("hello", "word"), (" the", "word"), ("a1", "other")])
def test_token_classes(text, cls):
    assert TC.token_class(text) == cls


def test_it_decides_nothing(setup):
    store, items = setup
    rep = TC.validity_report(store, items, policy=SUPERVISED_TARGET_V1)
    assert rep["decides"].startswith("nothing")
    assert all(isinstance(r, str) for r in rep["risks"])

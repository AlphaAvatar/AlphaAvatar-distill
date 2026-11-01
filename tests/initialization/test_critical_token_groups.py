"""Critical-token KL over GROUPS whose items carry different classes (StateEvalSuite.critical_groups).

One teacher's two generation modes are the case that needs it: a thinking-mode output closes a think block it
opened (think_close, final_answer, eos), a non-thinking one never opens one (final_answer, eos). A flat mean over
(group, class) pairs would weight the two modes 3:2 -- 60% and 40% -- by the count of their classes alone. With
groups declared, each group's classes are averaged inside the group and the groups weigh equally; without them the
historical flat mean, suite hash and protocol id are unchanged.
"""
from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from support.toy import TEACHER_GEOMETRY, build_tiny_model, make_items  # noqa: E402
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.planning.metrics import (  # noqa: E402
    CRITICAL_GROUP_AGGREGATION, StateEvaluator,
)
from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_UNCERTAINTY_WEIGHTED_KL, ScoringObjective,
)
from aadistill.initialization.scoring.positions import SUPERVISED_TARGET_V1  # noqa: E402
from aadistill.initialization.scoring.protocol_identity import ReductionSemantics  # noqa: E402
from aadistill.initialization.scoring.support import reference_topk_tail  # noqa: E402
from aadistill.initialization.specs.materialization import NumericalEnvironment  # noqa: E402
from aadistill.initialization.specs.metrics import (  # noqa: E402
    MeasurementError, StateEvalSuite, SuiteItem,
)

K = 8
CPU = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
OBJECTIVE = ScoringObjective("objective.toy_uncertainty_weighted_kl", 1, FORM_UNCERTAINTY_WEIGHTED_KL, entropy_top_k=K)
GROUPS = {"T": ("close", "answer", "eos"), "N": ("answer", "eos")}
TAGS = ("close", "answer", "eos")


def _items(groups=True, drop_class=None):
    items = []
    for k, raw in enumerate(make_items(n_per_subtype=4, seed=303)):
        mode = "T" if k % 2 == 0 else "N"
        ids = raw["input_ids"]
        n = int(ids.shape[1]) - 1
        pos = torch.arange(n)
        tags = {"answer": pos == n - 3, "eos": pos == n - 1,
                "close": (pos == n - 6) if mode == "T" else torch.zeros(n, dtype=torch.bool),
                "assistant": pos >= 4}
        if drop_class and mode == drop_class[0]:
            tags[drop_class[1]] = torch.zeros(n, dtype=torch.bool)
        items.append(SuiteItem(item_id=f"{raw['item_id']}|{mode}", input_ids=ids, domain=raw["domain"],
                               subtype=f"{raw['subtype']}|{mode}", tags=tags,
                               critical_group=mode if groups else None))
    return items


def _suite(groups=True):
    return StateEvalSuite(suite_id="test.groups", version=1, domains=("general", "math"),
                          subtypes={"general": ("text|T", "text|N"), "math": ("arith|T", "arith|N")},
                          critical_tags=TAGS, general_domain="general",
                          critical_groups=GROUPS if groups else None)


def _evaluator(suite, items, objective=None):
    return StateEvaluator(suite, items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
                          execution=ExecutionConfig(micro_batch_size=1, calibration_batch_packing="original_order_v1",
                                                    prediction_rows="scored_v1"),
                          distribution_support=reference_topk_tail(K), numerics=CPU, suite_content_sha256="toy",
                          scoring_objective=objective)


@pytest.fixture(scope="module")
def models():
    t = build_tiny_model(dict(TEACHER_GEOMETRY))
    t.config.use_cache = False
    c = build_tiny_model(dict(TEACHER_GEOMETRY), seed=29)
    c.config.use_cache = False
    return t, c


def _measure(models, groups=True, objective=None):
    ev = _evaluator(_suite(groups), _items(groups), objective)
    ev.prime_reference(models[0])
    return ev, ev.evaluate(models[1], "candidate")


def test_each_group_weighs_half_whatever_its_number_of_classes(models):
    _, r = _measure(models)
    v = r.values
    g = {m: v[f"state.critical_token_kl.group.{m}"] for m in GROUPS}
    for m, classes in GROUPS.items():
        assert g[m] == pytest.approx(sum(v[f"state.critical_token_kl.group.{m}.{t}"] for t in classes) / len(classes))
    assert v["state.critical_token_kl"] == pytest.approx(0.5 * g["T"] + 0.5 * g["N"])
    # not vacuous: the flat mean over (group, class) pairs is the 60:40 weighting, and it differs here
    flat = sum(v[f"state.critical_token_kl.group.{m}.{t}"] for m, c in GROUPS.items() for t in c) / 5
    assert g["T"] != pytest.approx(g["N"])
    assert flat == pytest.approx(0.6 * g["T"] + 0.4 * g["N"])
    assert v["state.critical_token_kl"] != pytest.approx(flat)
    assert r.detail["critical_aggregation"] == CRITICAL_GROUP_AGGREGATION


def test_the_objective_path_aggregates_groups_the_same_way(models):
    _, r = _measure(models, objective=OBJECTIVE)
    v = r.values
    assert v["state.objective.raw_kl.critical_token"] == pytest.approx(v["state.critical_token_kl"], rel=1e-6)
    g = {m: v[f"state.objective.critical_token.group.{m}"] for m in GROUPS}
    assert v["state.objective.critical_token"] == pytest.approx(0.5 * g["T"] + 0.5 * g["N"])


def test_a_group_declaration_is_a_new_protocol_and_leaves_the_historical_one_unchanged(models):
    grouped, flat = _evaluator(_suite(True), _items(True)), _evaluator(_suite(False), _items(False))
    assert grouped.measurement_protocol_id != flat.measurement_protocol_id
    assert grouped.reduction.as_dict()["critical_aggregation"] == CRITICAL_GROUP_AGGREGATION
    assert "critical_aggregation" not in flat.reduction.as_dict()
    assert flat.reduction.as_dict() == ReductionSemantics(
        chunk=flat.reduction.chunk, reference_strategy=flat.reduction.reference_strategy,
        distribution_support=flat.reduction.distribution_support).as_dict()
    s = _suite(False)
    historical = hashlib.sha256(json.dumps({
        "suite_id": s.suite_id, "version": s.version, "domains": list(s.domains),
        "subtypes": {k: list(v) for k, v in sorted(s.subtypes.items())},
        "critical_tags": list(s.critical_tags), "content_sha256": s.content_sha256,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert s.suite_hash == historical and _suite(True).suite_hash != historical
    _, r = _measure(models, groups=False)
    assert "critical_aggregation" not in r.detail
    assert not any(k.startswith("state.critical_token_kl.group.") for k in r.values)


def test_a_missing_class_an_ungrouped_item_or_an_undeclared_group_is_refused():
    with pytest.raises(MeasurementError, match=r"no scored position in their group \{'T': \['close'\]\}"):
        _evaluator(_suite(), _items(drop_class=("T", "close")))
    items = _items()
    items[0].critical_group = None
    with pytest.raises(MeasurementError, match="undeclared critical groups"):
        _evaluator(_suite(), items)
    items = _items()
    items[1].critical_group = "X"
    with pytest.raises(MeasurementError, match="undeclared critical groups"):
        _evaluator(_suite(), items)
    with pytest.raises(MeasurementError, match="not declared critical tags"):
        StateEvalSuite(suite_id="x", version=1, domains=("d",), subtypes={"d": ("s",)}, critical_tags=("eos",),
                       critical_groups={"T": ("close",)})


def test_the_shared_loader_carries_groups_and_each_items_group(tmp_path):
    from shared.evaluation.load_state_eval import load
    ids = list(range(12))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "suite_id": "toy", "version": 1, "domains": {"d": ["s|T", "s|N"]}, "critical_tags": list(TAGS),
        "critical_groups": {k: list(v) for k, v in GROUPS.items()}, "content_sha256": "c" * 64}))
    rows = [{"item_id": f"i{m}", "ids": ids, "domain": "d", "subtype": f"s|{m}", "n_prediction_positions": 11,
             "tags": {"eos": [10]}, "critical_group": m} for m in ("T", "N")]
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    suite, items, _ = load(tmp_path)
    assert suite.critical_groups == GROUPS and [i.critical_group for i in items] == ["T", "N"]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "suite_id": "toy", "version": 1, "domains": {"d": ["s|T", "s|N"]}, "critical_tags": list(TAGS),
        "content_sha256": "c" * 64}))
    suite, items, _ = load(tmp_path)
    assert suite.critical_groups is None

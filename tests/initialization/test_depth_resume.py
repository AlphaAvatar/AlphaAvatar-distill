"""A resumed DEPTH search computes the state a whole one does.

`depth.causal_kl_greedy_v1` accepts an earlier invocation's committed round records
(`OperatorContext.resume_state`): it checks them against the greedy rule, replays them without re-scoring, re-scores
the last replayed round's chosen subset (exact equality by default), and continues against the same intact-parent
reference. On a tiny genuine Qwen3, at both scoring paths (per candidate and prefix reuse):

1. 6 -> 2 resumed from a 6 -> 4 run's two rounds gives the whole 6 -> 2 run's removal order, kept layers, trace and
   child bytes, and its later rounds' tables bit for bit;
2. a record whose score this scorer does not reproduce, a record that breaks the tie rule, or one from another
   parent's numbering is refused before anything is built;
3. every other operator refuses a resume state, and the state stays out of every config and identity.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from aadistill.initialization.calibration.items import prepare_calibration_items  # noqa: E402
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    ContractViolation, OperatorContext, OperatorError,
)
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.scoring.positions import SUPERVISED_TARGET_V1, policy_config  # noqa: E402
from aadistill.initialization.scoring.support import reference_topk_tail  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.initialization.statistics.contribution import resumed_round_problems  # noqa: E402

from support.toy import build_tiny_model  # noqa: E402

GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, vocab_size=64, tie_word_embeddings=True)


@pytest.fixture(scope="module", autouse=True)
def _registered():
    from aadistill.initialization.adapters import register_builtin_adapters
    register_builtin_adapters()
    yield


@pytest.fixture(scope="module")
def model():
    m = build_tiny_model(GEOMETRY)
    m.config.use_cache = False
    return m


def _items():
    g = torch.Generator().manual_seed(7)
    out = []
    for i in range(6):
        length = 12 + 3 * i
        ids = torch.randint(1, 64, (1, length), generator=g)
        out.append({"item_id": f"item-{i}", "input_ids": ids, "domain": ("general", "math", "logic")[i % 3],
                    "subtype": ("text", "arith", "rules")[i % 3], "tags": {"assistant": list(range(3, length - 1))}})
    return prepare_calibration_items(out, profile_id="toy")


def _ctx(model, keep: int, reuse: bool, resume=None):
    items = _items()
    support = reference_topk_tail(8)
    config = {"n_calibration_items": len(items), **policy_config(SUPERVISED_TARGET_V1),
              "distribution_support": support.as_dict()}
    return OperatorContext(
        adapter=get_adapter("qwen3"), model=model, parent_spec=ArchSpec.of("qwen3", GEOMETRY),
        target_spec=ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": keep}), profile=None,
        calibration_items=items, seed=0, device="cpu", config=config, distribution_support=support,
        execution=ExecutionConfig(micro_batch_size=2, calibration_batch_packing="length_sorted_v1",
                                  prediction_rows="scored_v1", depth_prefix_reuse=reuse),
        position_policy=SUPERVISED_TARGET_V1, resume_state=resume)


def _run(model, keep, reuse, resume=None):
    return causal_kl_greedy.DepthCausalKLGreedyV1().execute(_ctx(model, keep, reuse, resume))


def _state_dict_equal(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize("reuse", [False, True], ids=["per_candidate", "prefix_reuse"])
def test_a_resumed_search_is_the_whole_search(model, reuse):
    whole = _run(model, 2, reuse)
    first = _run(model, 4, reuse)                     #: the earlier invocation: two rounds committed
    rounds = first.artifacts["search_rounds"]
    assert [r["chosen"] for r in rounds] == whole.trace["removal_order"][:2]
    resumed = _run(model, 2, reuse, {"rounds": copy.deepcopy(rounds), "source": "toy 6 -> 4"})
    assert resumed.trace == whole.trace                #: kept layers, removal order, everything identity reads
    assert resumed.local_metrics.values == whole.local_metrics.values
    assert _state_dict_equal(resumed.model, whole.model)
    #: the rounds after the resume were scored here, and equal the whole run's tables bit for bit
    later = [r for r in resumed.artifacts["search_rounds"] if not r.get("resumed")]
    assert [r["table"] for r in later] == [r["table"] for r in whole.artifacts["search_rounds"][2:]]
    ev = resumed.artifacts["resume"]
    assert ev["rounds_replayed"] == 2 and ev["rescored"]["abs_delta"] == 0.0
    assert resumed.artifacts["timing"]["candidate_subsets"] < whole.artifacts["timing"]["candidate_subsets"]


def test_records_this_scorer_does_not_reproduce_are_refused(model):
    rounds = _run(model, 4, True).artifacts["search_rounds"]
    bad = copy.deepcopy(rounds)
    bad[-1]["chosen_score"] += 1e-9                    #: a different scorer's number
    for t in bad[-1]["table"]:
        if t["candidate"] == bad[-1]["chosen"]:
            t["score"] = bad[-1]["chosen_score"]
    with pytest.raises(OperatorError, match="not written by this scorer"):
        _run(model, 2, True, {"rounds": bad})


def test_records_that_break_the_greedy_rule_are_refused(model):
    rounds = _run(model, 4, True).artifacts["search_rounds"]
    tie = copy.deepcopy(rounds)
    lo = min(t["candidate"] for t in tie[0]["table"] if t["candidate"] != tie[0]["chosen"])
    for t in tie[0]["table"]:
        if t["candidate"] == lo:
            t["score"] = tie[0]["chosen_score"]        #: a tie the record broke to the higher index
    if lo < tie[0]["chosen"]:
        assert any("ties to the lower index" in p for p in resumed_round_problems(tie, 6))
        with pytest.raises(OperatorError, match="not greedy round records"):
            _run(model, 2, True, {"rounds": tie})
    other = copy.deepcopy(rounds)
    other[1]["removed_before"] = [other[0]["chosen"] + 1]
    assert any("removed_before" in p for p in resumed_round_problems(other, 6))
    assert any("not every surviving layer" in p for p in resumed_round_problems(rounds, 7))
    assert resumed_round_problems(rounds, 6) == []
    with pytest.raises(OperatorError, match="resumed rounds for a search of"):
        _run(model, 5, True, {"rounds": rounds})


def test_only_an_operator_that_can_continue_takes_a_resume_state(model):
    from aadistill.initialization.operators.ffn.dense.activation_importance import FFNActivationImportanceV0
    ctx = _ctx(model, 6, True, {"rounds": []})
    ctx.target_spec = ArchSpec.of("qwen3", {**GEOMETRY, "intermediate_size": 32})
    with pytest.raises(ContractViolation, match="cannot continue"):
        FFNActivationImportanceV0().execute(ctx)
    assert "resume_state" not in _ctx(model, 2, True, {"rounds": []}).config

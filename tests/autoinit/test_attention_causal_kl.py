"""`attention.causal_kl_v1` — the one-shot causal head scorer.

Every test here EXECUTES the operator. The C3 pilot's whole cost is forward
passes on a paid pod, so the cheap thing to be sure of first is that the code
path runs, that the ablation means what the module claims it means, and that
the batch size is read from the hashed step config rather than from the
process-wide runtime default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.profiles import NO_CALIBRATION  # noqa: E402
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators import get_implementation  # noqa: E402
from aadistill.initialization.operators.attention.gqa import causal_kl  # noqa: E402
from aadistill.initialization.operators.attention.gqa.causal_kl import (  # noqa: E402
    BATCH_SIZE_CONFIG_KEY,
    domain_subtype_map,
    head_ablated,
    resolve_forward_batch_size,
)
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorContext, OperatorError,
)
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402
from dataclasses import replace  # noqa: E402

from conftest import build_tiny_model  # noqa: E402

#: 8 query heads over 2 KV groups -> keep 2 of 4 within each group. The same
#: geometry the sibling ATTENTION operator's tests use, so a structural
#: difference between the two shows up as a difference here.
GEOMETRY = dict(hidden_size=32, num_hidden_layers=3, intermediate_size=48,
                num_attention_heads=8, num_key_value_heads=2, head_dim=8,
                vocab_size=128, tie_word_embeddings=True)
KEEP_HEADS = 4


def items(n_per_subtype=2, seq=12, vocab=128, seed=11):
    """Two domains and three sub-types, so domain balancing is exercised."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for subtype in ("general", "alpha", "beta"):
        for k in range(n_per_subtype):
            out.append({
                "item_id": f"{subtype}/{k}",
                "domain": "general" if subtype == "general" else "task",
                "subtype": subtype,
                "input_ids": torch.randint(0, vocab, (1, seq), generator=g),
            })
    return out


def ragged_items(vocab=128, seed=13):
    """Different lengths, so padding is real and the mask has to work."""
    g = torch.Generator().manual_seed(seed)
    lengths = {"general": (9, 14), "alpha": (11, 7), "beta": (16, 10)}
    out = []
    for subtype, lens in lengths.items():
        for k, length in enumerate(lens):
            out.append({
                "item_id": f"{subtype}/{k}",
                "domain": "general" if subtype == "general" else "task",
                "subtype": subtype,
                "input_ids": torch.randint(0, vocab, (1, length), generator=g),
            })
    return out


def context(model, parent_spec, target_spec, calib, *, batch_size=None,
            execution=None):
    config = {"n_calibration_items": len(calib)}
    if batch_size is not None:
        config[BATCH_SIZE_CONFIG_KEY] = batch_size
    kwargs = {}
    if execution is not None:
        kwargs["execution"] = execution
    return OperatorContext(
        adapter=QWEN3_ADAPTER, model=model, parent_spec=parent_spec,
        target_spec=target_spec, profile=NO_CALIBRATION,
        calibration_items=calib, seed=0, device="cpu", config=config, **kwargs)


@pytest.fixture(autouse=True)
def registered():
    """Explicit registration must not leak into an unrestricted BeamSearch."""
    causal_kl.register(replace=True)
    yield
    causal_kl.unregister()


@pytest.fixture
def geo():
    return ArchSpec.of("qwen3", GEOMETRY)


@pytest.fixture
def target(geo):
    return geo.replace(num_attention_heads=KEEP_HEADS)


def run(model, geo, target, calib, **kwargs):
    impl = get_implementation("attention.causal_kl_v1")
    return impl.execute(context(model, geo, target, calib, **kwargs))


# --- the claim the whole operator rests on ---------------------------------

def test_zeroing_the_o_proj_columns_equals_deleting_the_head():
    """Bit-exact, not approximately. No hooks, no surgery, no rebuilt module.

    If this were only approximately true the scores would be measuring the
    approximation as well as the head, and the cheapest ablation would have to
    be replaced by materializing a model per head.
    """
    model = build_tiny_model(GEOMETRY)
    ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(3))
    block = QWEN3_ADAPTER.blocks(model)[1]
    o_proj = QWEN3_ADAPTER.attention(block).o_proj
    head, head_dim = 5, GEOMETRY["head_dim"]

    with torch.no_grad():
        #: Real deletion: the head's activation never reaches `o_proj`, done by
        #: zeroing the head's OUTPUT rather than its weights. A different
        #: mechanism producing the same bits is the evidence; the same
        #: mechanism twice would be a tautology.
        captured = {}

        def kill_head(_module, _args, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.clone()
            out[..., head * head_dim:(head + 1) * head_dim] = 0
            captured["hit"] = True
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

        attn = QWEN3_ADAPTER.attention(block)
        inner = attn.o_proj

        class _Killed(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x):
                x = x.clone()
                x[..., head * head_dim:(head + 1) * head_dim] = 0
                captured["hit"] = True
                return self.wrapped(x)

        attn.o_proj = _Killed(inner)
        deleted = model(ids).logits.clone()
        attn.o_proj = inner
    assert captured.get("hit"), "the deletion path never ran"

    with torch.no_grad():
        with head_ablated(o_proj, head, head_dim):
            zeroed = model(ids).logits.clone()

    assert torch.equal(deleted, zeroed), (
        "zeroing the column block is not equivalent to removing the head")


def test_the_columns_are_restored_even_when_the_body_raises():
    """A scorer that leaves the model damaged corrupts every later step."""
    model = build_tiny_model(GEOMETRY)
    o_proj = QWEN3_ADAPTER.attention(QWEN3_ADAPTER.blocks(model)[0]).o_proj
    before = o_proj.weight.detach().clone()
    with pytest.raises(RuntimeError, match="deliberate"):
        with head_ablated(o_proj, 2, GEOMETRY["head_dim"]):
            assert not torch.equal(o_proj.weight, before), "nothing was zeroed"
            raise RuntimeError("deliberate")
    assert torch.equal(o_proj.weight, before)


def test_the_model_is_unchanged_after_a_whole_scoring_run(geo, target):
    """896 ablations on the pod; every one of them has to put its columns back."""
    model = build_tiny_model(GEOMETRY)
    before = [p.detach().clone() for p in model.parameters()]
    run(model, geo, target, items())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_the_parent_config_is_not_mutated_either(geo, target):
    """Not only the weights. `depth.causal_kl_greedy_v1` sets
    `model.config.use_cache = False`, which reaches the CHILD's config.json
    and so its `config_sha256` -- an identity that depends on whether an
    operator ran. This one passes the flag per forward instead."""
    model = build_tiny_model(GEOMETRY)
    before = model.config.to_dict()
    run(model, geo, target, items())
    assert model.config.to_dict() == before


def test_the_source_contains_no_use_cache_assignment():
    """The repository audits the set of modules that flip this bit; this
    operator must stay out of that set."""
    import re

    src = (REPO / "src/aadistill/initialization/operators/attention/gqa/"
           "causal_kl.py").read_text()
    for line in src.splitlines():
        assert not re.match(r"^[A-Za-z_][\w.]*\.use_cache\s*=\s*False\s*(#.*)?$",
                            line.strip()), line


def test_the_forwards_do_pass_use_cache_false():
    """The per-call alternative is actually used, not merely not-assigned."""
    import ast
    import inspect

    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    for fn in (m._forward_block, m._forward_item):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        kwargs = {k.arg for c in calls for k in c.keywords}
        assert "use_cache" in kwargs, f"{fn.__name__} does not pass use_cache"


# --- the deadline, and staying visible ------------------------------------


class _Deadline:
    """`OperatorContext.deadline`'s contract: `.check(what)` raises or returns.

    Modelled on `C1OperatorDeadline`, which raises `C1DriverError` once the
    session's soft stop is reached.
    """

    def __init__(self, allow: int):
        self.allow, self.calls = allow, []

    def check(self, what: str) -> None:
        self.calls.append(what)
        if len(self.calls) > self.allow:
            raise RuntimeError(f"budget exhausted: {what}")


def test_the_deadline_stops_the_scorer(geo, target):
    """897 corpus passes is tens of minutes to hours. Without this the only
    thing between an overrun and the cost watchdog is nothing -- which is how
    DEPTH ran 10.78 h against a 3.0 h budget."""
    model = build_tiny_model(GEOMETRY)
    deadline = _Deadline(allow=2)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        impl = get_implementation("attention.causal_kl_v1")
        ctx = context(model, geo, target, items())
        impl.execute(replace(ctx, deadline=deadline))
    assert len(deadline.calls) == 3, "it kept going after the refusal"


def test_a_stop_still_leaves_the_model_intact(geo, target):
    """The `finally` has to survive the exception, not only the happy path."""
    model = build_tiny_model(GEOMETRY)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError):
        impl = get_implementation("attention.causal_kl_v1")
        ctx = context(model, geo, target, items())
        impl.execute(replace(ctx, deadline=_Deadline(allow=1)))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_the_deadline_message_says_where_it_was(geo, target):
    """A stop that cannot say how far it got is a stop nobody can price."""
    model = build_tiny_model(GEOMETRY)
    deadline = _Deadline(allow=10_000)
    impl = get_implementation("attention.causal_kl_v1")
    impl.execute(replace(context(model, geo, target, items()),
                         deadline=deadline))
    assert deadline.calls, "the deadline was never consulted"
    assert all("attention.causal_kl_v1" in c for c in deadline.calls)
    assert "forwards done" in deadline.calls[-1]
    #: One check per (group, layer), not per ablation: 896 checks and 896
    #: printed lines would bury the signal they exist to produce.
    n_groups, n_layers = len(items()), GEOMETRY["num_hidden_layers"]
    assert len(deadline.calls) == n_groups * n_layers


def test_no_deadline_is_still_allowed(geo, target):
    """`ctx.deadline` is None on every path that does not budget one."""
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    assert out.trace["seconds"] >= 0


def test_the_scorer_clock_is_bracketed_by_cuda_syncs():
    """STRUCTURAL, because this dev box has no CUDA to measure it on.

    CUDA kernels are asynchronous: without a synchronize the timer stops
    while work is still queued, and the tail lands on whatever runs next. The
    adoption gate is a RATIO of two such clocks, so a mis-attributed tail
    moves the verdict — and a CPU test cannot tell the difference, which is
    exactly why the shape is asserted rather than the number.
    """
    import ast
    import inspect

    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    import textwrap

    #: `dedent`, not `lstrip`: the latter strips only the FIRST line, so a
    #: method's body stays indented and `ast.parse` raises IndentationError.
    src = textwrap.dedent(inspect.getsource(m.AttentionCausalKLV1.apply))
    tree = ast.parse(src)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_cuda_sync":
            lines.append(node.lineno)
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "started" for t in node.targets)):
            start_line = node.lineno
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "scorer_seconds"
                        for t in node.targets)):
            stop_line = node.lineno
    assert len(lines) >= 2, "the scorer clock is not synchronized at both ends"
    assert any(l < start_line for l in lines), "no sync before the timer starts"
    assert any(start_line < l < stop_line for l in lines), (
        "no sync before the timer stops")


def test_the_warm_up_contributes_no_causal_evidence(geo, target):
    """It ablates nothing and its logits are discarded; it exists only so the
    first TIMED forward is not paying for allocator growth and autotuning
    that would otherwise land entirely on whichever arm ran first."""
    from aadistill.initialization.operators.attention.gqa.causal_kl import warm_up

    model = build_tiny_model(GEOMETRY)
    before = [p.detach().clone() for p in model.parameters()]
    info = warm_up(model, items(), "cpu", batch_size=4, n=2)
    assert info["warmup_forwards"] >= 1 and info["warmup_items"] >= 1
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    #: And it changes nothing about the scores that follow.
    a = run(build_tiny_model(GEOMETRY), geo, target, items())
    warm_up(model, items(), "cpu", batch_size=1, n=2)
    b = run(build_tiny_model(GEOMETRY), geo, target, items())
    assert a.artifacts["kept_heads"] == b.artifacts["kept_heads"]


def test_the_warm_up_is_a_no_op_when_asked_for_nothing():
    from aadistill.initialization.operators.attention.gqa.causal_kl import warm_up

    model = build_tiny_model(GEOMETRY)
    assert warm_up(model, [], "cpu")["warmup_forwards"] == 0
    assert warm_up(model, items(), "cpu", n=0)["warmup_forwards"] == 0


def test_the_scorer_clock_excludes_the_child_build(geo, target):
    """The gate compares the SCORER; aggregation, selection and the child
    build are identical in both arms and would dilute the ratio."""
    out = run(build_tiny_model(GEOMETRY), geo, target, items())
    assert 0 < out.trace["scorer_seconds"] <= out.trace["seconds"]


def test_progress_is_printed_and_bounded(geo, target, capfd):
    model = build_tiny_model(GEOMETRY)
    run(model, geo, target, items())
    lines = [ln for ln in capfd.readouterr().out.splitlines()
             if ln.startswith("attention.causal_kl_v1:")]
    assert lines, "the scorer ran silently"
    assert len(lines) == len(items()) * GEOMETRY["num_hidden_layers"]
    assert "forwards" in lines[-1] and "fwd/min" in lines[-1]


# --- structural contract, identical to the other ATTENTION operators -------

def test_it_changes_only_the_query_head_count(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    after = QWEN3_ADAPTER.spec_of(out.model)
    assert geo.diff(after) == frozenset({"num_attention_heads"})
    assert after["num_attention_heads"] == KEEP_HEADS


def test_gqa_grouping_kv_heads_and_rope_are_preserved(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    a, b = model.config, out.model.config
    assert b.num_key_value_heads == a.num_key_value_heads == 2
    assert b.head_dim == a.head_dim == 8
    assert b.rope_parameters == a.rope_parameters
    for src, dst in zip(QWEN3_ADAPTER.blocks(model), QWEN3_ADAPTER.blocks(out.model)):
        s, d = QWEN3_ADAPTER.attention(src), QWEN3_ADAPTER.attention(dst)
        assert torch.equal(s.k_proj.weight, d.k_proj.weight)
        assert torch.equal(s.v_proj.weight, d.v_proj.weight)


def test_two_of_four_heads_survive_in_each_gqa_group(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    per_group = KEEP_HEADS // GEOMETRY["num_key_value_heads"]
    group_size = GEOMETRY["num_attention_heads"] // GEOMETRY["num_key_value_heads"]
    for kept in out.artifacts["kept_heads"]:
        assert kept == sorted(kept) and len(kept) == KEEP_HEADS
        for g in range(GEOMETRY["num_key_value_heads"]):
            lo, hi = g * group_size, (g + 1) * group_size
            assert len([h for h in kept if lo <= h < hi]) == per_group


def test_the_kept_q_rows_are_copied_verbatim(geo, target):
    """Selection, not re-derivation: a kept head's weights must be untouched."""
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    d = GEOMETRY["head_dim"]
    for kept, src, dst in zip(out.artifacts["kept_heads"],
                              QWEN3_ADAPTER.blocks(model),
                              QWEN3_ADAPTER.blocks(out.model)):
        s, t = QWEN3_ADAPTER.attention(src), QWEN3_ADAPTER.attention(dst)
        for new, old in enumerate(kept):
            assert torch.equal(t.q_proj.weight[new * d:(new + 1) * d],
                               s.q_proj.weight[old * d:(old + 1) * d])
            assert torch.equal(t.o_proj.weight[:, new * d:(new + 1) * d],
                               s.o_proj.weight[:, old * d:(old + 1) * d])


# --- the signal is causal, and it is not the sibling's signal --------------

def test_a_head_that_does_nothing_scores_lowest(geo, target):
    """Ground truth, planted: a head whose write is already zero cannot be
    causal, so ablating it must change nothing and it must be dropped."""
    model = build_tiny_model(GEOMETRY)
    d = GEOMETRY["head_dim"]
    dead = 1
    with torch.no_grad():
        for block in QWEN3_ADAPTER.blocks(model):
            QWEN3_ADAPTER.attention(block).o_proj.weight[:, dead * d:(dead + 1) * d] = 0
    out = run(model, geo, target, items())
    for kept in out.artifacts["kept_heads"]:
        assert dead not in kept, "a head with no causal effect was retained"


def test_the_retained_share_is_reported_per_layer(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    shares = out.local_metrics.detail["per_layer_retained_share"]
    assert len(shares) == GEOMETRY["num_hidden_layers"]
    assert all(0.0 <= s <= 1.0 for s in shares)
    assert out.local_metrics.values["op.attention.retained_causal_kl_min"] == min(shares)


# --- the evidence: the landscape, not only the decision -------------------

def _evidence(out):
    return out.artifacts["causal_head_evidence"]


def test_the_per_item_landscape_is_retained(geo, target):
    """A head map cannot distinguish a decisive margin from a tie."""
    calib = items()
    out = run(build_tiny_model(GEOMETRY), geo, target, calib)
    ev = _evidence(out)
    L, H, N = (GEOMETRY["num_hidden_layers"], GEOMETRY["num_attention_heads"],
               len(calib))
    assert len(ev["per_item_kl"]) == L
    assert all(len(layer) == H for layer in ev["per_item_kl"])
    assert all(len(head) == N for layer in ev["per_item_kl"] for head in layer)
    assert all(isinstance(v, float) for layer in ev["per_item_kl"]
               for head in layer for v in head)


def test_item_metadata_is_stored_once_as_columns(geo, target):
    """Not repeated 60k times. Columns, parallel to the value axis."""
    calib = items()
    ev = _evidence(run(build_tiny_model(GEOMETRY), geo, target, calib))
    assert ev["item_ids"] == [i["item_id"] for i in calib]
    assert ev["item_domains"] == [i["domain"] for i in calib]
    assert ev["item_subtypes"] == [i["subtype"] for i in calib]
    assert ev["item_prediction_positions"] == [
        int(i["input_ids"].shape[-1]) - 1 for i in calib]


def test_the_values_are_in_the_mixtures_own_order_not_the_batchers(geo, target):
    """B1 and B4 visit items in different GROUPINGS but the same ORDER; the
    evidence must be indexed by the mixture, or the two arms' landscapes
    could not be compared row by row."""
    calib = ragged_items()
    b1 = _evidence(run(build_tiny_model(GEOMETRY), geo, target, calib,
                       batch_size=1))
    b4 = _evidence(run(build_tiny_model(GEOMETRY), geo, target, calib,
                       batch_size=4))
    assert b1["item_ids"] == b4["item_ids"] == [i["item_id"] for i in calib]
    for l in range(GEOMETRY["num_hidden_layers"]):
        for h in range(GEOMETRY["num_attention_heads"]):
            for a, b in zip(b1["per_item_kl"][l][h], b4["per_item_kl"][l][h]):
                assert a == pytest.approx(b, abs=1e-6)


def test_the_aggregate_scores_are_retained_per_head(geo, target):
    out = run(build_tiny_model(GEOMETRY), geo, target, items())
    ev = _evidence(out)
    assert len(ev["head_scores"]) == GEOMETRY["num_hidden_layers"]
    assert all(len(s) == GEOMETRY["num_attention_heads"]
               for s in ev["head_scores"])


def test_each_gqa_group_records_its_cut(geo, target):
    out = run(build_tiny_model(GEOMETRY), geo, target, items())
    ev = _evidence(out)
    n_kv = GEOMETRY["num_key_value_heads"]
    per_group = KEEP_HEADS // n_kv
    for layer_decisions, kept in zip(ev["gqa_decisions"],
                                     out.artifacts["kept_heads"]):
        assert len(layer_decisions) == n_kv
        for d in layer_decisions:
            assert len(d["selected_heads"]) == per_group
            assert d["ranked_heads"][:per_group] == sorted(
                d["selected_heads"], key=lambda h: (
                    -d["scores"][h - d["member_heads"][0]], h))
            assert d["cutoff_selected"] >= d["cutoff_rejected"]
            assert d["cutoff_margin"] == pytest.approx(
                d["cutoff_selected"] - d["cutoff_rejected"])
        #: And the decisions agree with the map the operator actually built.
        assert sorted(h for d in layer_decisions
                      for h in d["selected_heads"]) == kept


def test_the_cut_is_consistent_with_the_shared_selector(geo, target):
    """The decision record reproduces `select_q_heads_by_score`, not a second
    ordering rule that could drift from it."""
    from aadistill.initialization.operators.attention.gqa._common import (
        select_q_heads_by_score)

    out = run(build_tiny_model(GEOMETRY), geo, target, items())
    ev = _evidence(out)
    for layer, scores in enumerate(ev["head_scores"]):
        expected = select_q_heads_by_score(
            scores, GEOMETRY["num_attention_heads"],
            GEOMETRY["num_key_value_heads"], KEEP_HEADS)
        assert out.artifacts["kept_heads"][layer] == expected


def test_no_logits_are_retained(geo, target):
    """The evidence is scores, not 60k x 151936 floats."""
    ev = _evidence(run(build_tiny_model(GEOMETRY), geo, target, items()))
    assert "logits" not in json.dumps(ev)[:200].lower()
    assert set(ev) == {
        "score", "calibration_forward_batch_size", "item_ids", "item_domains",
        "item_subtypes", "item_prediction_positions", "per_item_kl",
        "head_scores", "gqa_decisions"}


def test_the_evidence_is_json_serializable(geo, target):
    """It has to reach a record, so it may hold no tensors and no NaN."""
    ev = _evidence(run(build_tiny_model(GEOMETRY), geo, target, items()))
    text = json.dumps(ev)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["head_scores"] == ev["head_scores"]


# --- the batch size is CONFIG, and nothing else ----------------------------

def test_the_batch_size_comes_from_the_step_config(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items(), batch_size=3)
    assert out.trace["calibration_forward_batch_size"] == 3


def test_an_undeclared_batch_size_is_one_not_the_process_default(geo, target):
    """THE COUPLING THIS OPERATOR MUST NOT HAVE.

    `DEFAULT_EXECUTION.micro_batch_size` is 4 on this branch. If the operator
    fell back to it, an identical step would produce a B4 result on this
    branch and a B1 result on another, with the same hash for both.
    """
    from aadistill.initialization.execution import DEFAULT_EXECUTION

    assert DEFAULT_EXECUTION.micro_batch_size != 1, (
        "this test is vacuous unless the process default differs from 1")
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    assert out.trace["calibration_forward_batch_size"] == 1


def test_ctx_execution_is_ignored_entirely(geo, target):
    """Set the runtime knob to something loud and show it does not reach in."""
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items(),
              execution=ExecutionConfig(micro_batch_size=7))
    assert out.trace["calibration_forward_batch_size"] == 1
    out = run(model, geo, target, items(), batch_size=2,
              execution=ExecutionConfig(micro_batch_size=7))
    assert out.trace["calibration_forward_batch_size"] == 2


def test_the_operator_source_never_reads_micro_batch_size():
    """Executable check, so a fallback cannot be reintroduced quietly."""
    import ast
    import inspect

    src = inspect.getsource(causal_kl)
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "micro_batch_size" not in attrs
    live = " ".join(n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in docstrings)
    assert "micro_batch_size" not in live


@pytest.mark.parametrize("bad", [0, -1])
def test_a_batch_size_below_one_is_refused(bad):
    with pytest.raises(OperatorError, match="must be >= 1"):
        resolve_forward_batch_size({BATCH_SIZE_CONFIG_KEY: bad})


@pytest.mark.parametrize("bad", ["4", 4.0, 4.7, True, False, None, [4]])
def test_a_non_int_batch_size_is_REFUSED_not_coerced(bad):
    """`int(value)` would accept every one of these.

    The field is hashed into the step identity, so `4`, `4.0`, `"4"` and
    `True` must not be allowed to mean the same execution while serializing to
    four different states -- a replay would then disagree with its own record
    about what it ran. Same rule as `ExecutionConfig`, for the same reason.
    """
    with pytest.raises(OperatorError, match="must be an int"):
        resolve_forward_batch_size({BATCH_SIZE_CONFIG_KEY: bad})


def test_the_resolver_does_not_call_int():
    """Executable check: a future edit cannot reintroduce the coercion."""
    import ast
    import inspect

    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    tree = ast.parse(inspect.getsource(m.resolve_forward_batch_size).lstrip())
    calls = {getattr(n.func, "id", "") for n in ast.walk(tree)
             if isinstance(n, ast.Call)}
    assert "int" not in calls, "the identity-bearing field is being coerced"


def test_the_pilot_step_constructor_refuses_the_same_values():
    """The pilot must not build a step the operator will later refuse."""
    from experiments.phase_c3.pilot import causal_step

    for bad in ("4", 4.0, True, None):
        with pytest.raises((TypeError, ValueError)):
            causal_step(bad)


# --- a non-finite score is a FAILED scorer, never a zero ------------------


class _Poison:
    """Makes one forward return non-finite logits, then restores itself."""

    def __init__(self, model, adapter, layer=1):
        self.proj = adapter.attention(adapter.blocks(model)[layer]).o_proj
        self.saved = None

    def __enter__(self):
        self.saved = self.proj.weight.detach().clone()
        with torch.no_grad():
            self.proj.weight.fill_(float("inf"))
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            self.proj.weight.copy_(self.saved)
        return False


@pytest.mark.parametrize("batch_size", [1, 4])
def test_a_non_finite_kl_refuses_rather_than_scoring_zero(geo, target,
                                                          batch_size):
    """60k forwards feed a `sorted()` that materializes a head map. One NaN
    sorts to an arbitrary position and the map is a fiction."""
    model = build_tiny_model(GEOMETRY)
    with _Poison(model, QWEN3_ADAPTER):
        with pytest.raises(OperatorError, match="non-finite causal KL"):
            run(model, geo, target, items(), batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_the_refusal_names_where_it_happened(geo, target, batch_size):
    model = build_tiny_model(GEOMETRY)
    with _Poison(model, QWEN3_ADAPTER):
        with pytest.raises(OperatorError) as exc:
            run(model, geo, target, items(), batch_size=batch_size)
    text = str(exc.value)
    assert "layer" in text and "head" in text
    assert (f"B{batch_size}" in text) or ("B1" in text and batch_size == 1)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_the_model_is_fully_restored_after_a_refusal(geo, target, batch_size):
    """The `finally` in `head_ablated` must survive this exception too."""
    model = build_tiny_model(GEOMETRY)
    poison = _Poison(model, QWEN3_ADAPTER)
    with poison:
        before = [p.detach().clone() for p in model.parameters()]
        with pytest.raises(OperatorError):
            run(model, geo, target, items(), batch_size=batch_size)
        assert all(torch.equal(a, b)
                   for a, b in zip(before, model.parameters()))


def test_a_non_finite_aggregate_refuses_before_materialization(geo, target,
                                                               monkeypatch):
    """The per-item guard cannot catch a mean that overflows, and the
    aggregate is the value `sorted()` actually reads."""
    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    calls = {"n": 0}

    def poisoned(means, domains):
        calls["n"] += 1
        return (float("nan") if calls["n"] == 5 else 1.0), {}

    monkeypatch.setattr(m, "domain_balanced_score", poisoned)
    model = build_tiny_model(GEOMETRY)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(OperatorError, match="aggregate score"):
        run(model, geo, target, items())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_a_missing_item_value_refuses(geo, target, monkeypatch):
    """Incomplete evidence must not be aggregated into a confident score."""
    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    real = m.micro_batches

    def short(items_, batch_size, **kw):
        groups = list(real(items_, batch_size, **kw))
        return iter(groups[:-1])          # one group never scored

    monkeypatch.setattr(m, "micro_batches", short)
    with pytest.raises(OperatorError, match="produced no causal KL"):
        run(build_tiny_model(GEOMETRY), geo, target, items())


def test_nothing_is_clamped_or_substituted():
    """Stated as source, because a clamp would pass every test above."""
    import inspect

    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    src = inspect.getsource(m)
    for banned in ("nan_to_num", "torch.clamp", "= 0.0 if", "or 0.0"):
        assert banned not in src, f"the scorer appears to substitute: {banned}"


def test_resolve_accepts_a_missing_config():
    assert resolve_forward_batch_size(None) == 1
    assert resolve_forward_batch_size({}) == 1


# --- B1 vs B4 on CPU, which is the pilot's question asked for free ---------

def test_b1_and_b4_agree_on_equal_length_items(geo, target):
    """No padding, fp32, CPU: the two should agree to floating-point noise.

    This is NOT the pilot's result — the pilot asks about bf16 on CUDA over
    ragged real items, where padded batching is already MEASURED to move
    decisions. What it does establish is that the batched path is not simply
    wrong, which is the failure a paid pod should not be the first to find.
    """
    calib = items(seq=12)
    b1 = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=1)
    b4 = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=4)
    assert b1.artifacts["kept_heads"] == b4.artifacts["kept_heads"]
    for x, y in zip(b1.local_metrics.detail["per_layer_retained_share"],
                    b4.local_metrics.detail["per_layer_retained_share"]):
        assert x == pytest.approx(y, abs=1e-6)


def test_ragged_items_are_scored_over_their_own_positions(geo, target):
    """Padding must not leak: B1 and B4 over RAGGED items still have to agree
    on CPU fp32, because the mask is what makes a padded batch a per-item
    measurement rather than a pooled one."""
    calib = ragged_items()
    b1 = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=1)
    b4 = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=4)
    assert b1.artifacts["kept_heads"] == b4.artifacts["kept_heads"]
    for x, y in zip(b1.local_metrics.detail["per_layer_retained_share"],
                    b4.local_metrics.detail["per_layer_retained_share"]):
        assert x == pytest.approx(y, abs=1e-6)


def test_the_trace_reports_padding_separately_from_work(geo, target):
    """The pilot must be able to say how much of B4's work was padding."""
    calib = ragged_items()
    out = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=4)
    valid = sum(int(i["input_ids"].shape[-1]) for i in calib)
    assert out.trace["valid_tokens"] == valid
    assert out.trace["padded_positions"] > 0, "ragged items must pad"
    b1 = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=1)
    assert b1.trace["padded_positions"] == 0


def test_physical_invocations_fall_with_batch_size_but_the_work_does_not(geo, target):
    """The distinction the pricing rests on, measured rather than assumed."""
    calib = items()
    b1 = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=1)
    b4 = run(build_tiny_model(GEOMETRY), geo, target, calib, batch_size=4)
    assert b4.trace["physical_forward_invocations"] < \
        b1.trace["physical_forward_invocations"]
    assert b1.trace["item_forward_equivalents"] == \
        b4.trace["item_forward_equivalents"]


# --- the plan prices algorithmic work, identically in both arms ------------

def test_the_plan_counts_one_ablation_per_head_plus_one_reference(geo, target):
    impl = get_implementation("attention.causal_kl_v1")
    plan = impl.plan(geo, target, QWEN3_ADAPTER, {"n_calibration_items": 67})
    ablations = GEOMETRY["num_hidden_layers"] * GEOMETRY["num_attention_heads"]
    assert plan.forward_passes == (ablations + 1) * 67
    assert plan.stats_passes == 0


def test_both_arms_price_identically(geo, target):
    """A B4 plan at a quarter of B1 would ASSERT the speedup being measured."""
    impl = get_implementation("attention.causal_kl_v1")
    b1 = impl.plan(geo, target, QWEN3_ADAPTER,
                   {"n_calibration_items": 67, BATCH_SIZE_CONFIG_KEY: 1})
    b4 = impl.plan(geo, target, QWEN3_ADAPTER,
                   {"n_calibration_items": 67, BATCH_SIZE_CONFIG_KEY: 4})
    assert b1.forward_passes == b4.forward_passes


def test_the_plan_is_priced_above_zero_by_the_generic_cost_model(geo, target):
    """The defect the pricing repair closed, asserted through this operator."""
    from aadistill.runtime.cost import HardwareProfile, operator_cost

    impl = get_implementation("attention.causal_kl_v1")
    hardware = HardwareProfile(name="synthetic", price_per_hour_usd=1.0,
                               effective_tflops=100.0, vram_gb=48.0,
                               measured=False, source="test")
    cost = operator_cost(impl, geo, target, QWEN3_ADAPTER,
                         calibration_tokens=59_830, seq_len=1024,
                         hardware=hardware)
    assert cost.gpu_seconds > 0


def test_a_target_that_removes_nothing_is_refused(geo):
    impl = get_implementation("attention.causal_kl_v1")
    with pytest.raises(OperatorError, match="nothing to remove"):
        impl.plan(geo, geo, QWEN3_ADAPTER, {"n_calibration_items": 1})


# --- registration is a decision, never an import ---------------------------

def test_importing_the_module_registers_nothing():
    """`BeamSearch` falls back to the whole registry, so an import-time
    registration adds a branch to searches that never asked for one."""
    import subprocess

    probe = (
        "import sys; sys.path.insert(0, 'src');"
        "import aadistill.initialization.operators.attention.gqa.causal_kl;"
        "from aadistill.initialization.operators.base import registered_implementations;"
        "print('attention.causal_kl_v1' in dict(registered_implementations()))"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False", done.stdout


def test_it_is_absent_from_the_builtin_operators():
    from aadistill.initialization.operators.register import BUILTIN_OPERATORS

    assert "attention.causal_kl_v1" not in {i.impl_id for i in BUILTIN_OPERATORS}


# --- the duplicated domain map may not drift -------------------------------

def test_the_domain_map_matches_the_depth_operators_private_copy():
    """The duplication is deliberate and bounded; this is what bounds it."""
    from aadistill.initialization.operators.depth.causal_kl_greedy import _domain_map

    for calib in (items(), ragged_items()):
        assert domain_subtype_map(calib) == _domain_map(calib)


def test_the_domain_map_agrees_on_the_real_frozen_mixture():
    """Not only on synthetic items shaped like the real ones."""
    from aadistill.initialization.operators.depth.causal_kl_greedy import _domain_map

    #: The frozen 67-item `calib.domain_balanced@v1` mixture itself.
    #:
    #: `artifacts/` is gitignored, so on a fresh clone these bytes are simply
    #: absent and an unconditional assertion would be red for everyone who has
    #: not built them. The skip is therefore on the TREE and the assertion on
    #: the FILE: where the artifact root exists, a missing mixture is a real
    #: failure rather than a quiet absence.
    if not (REPO / "artifacts/stage1").is_dir():
        pytest.skip("artifacts/ is gitignored and not built in this tree")
    manifest = REPO / "artifacts/stage1/e8_calibration_v1/items.jsonl"
    assert manifest.is_file(), f"{manifest} is missing from a built artifact tree"
    import json

    real = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    assert len(real) == 67, f"expected the frozen 67-item mixture, got {len(real)}"
    assert domain_subtype_map(real) == _domain_map(real)

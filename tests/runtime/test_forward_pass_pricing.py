"""A plan that declares model forwards must be priced for them.

`operator_cost` had two arms: a branch naming `depth.causal_kl_greedy_v1` by
id, and a branch for `plan.stats_passes`. An operator declaring
`forward_passes` and no `stats_passes` matched neither and fell through to
`flops = 0` -- so a plan declaring tens of thousands of model forwards priced
at exactly zero GPU seconds.

The hole stayed invisible because the only forward-heavy operator in the
library is the one named in the first branch. The first implementation to walk
into it would be a NEW one, arriving with a plan nobody had priced, which is
precisely what `attention.causal_kl_v1` is.

These are SYNTHETIC. They define a throwaway implementation rather than
importing a real one, so they test the generic contract and cannot be satisfied
by special-casing another id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.adapters import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.profiles import CalibrationNeed  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorImplementation, OperatorPlan,
)
from aadistill.runtime.cost import HardwareProfile, operator_cost  # noqa: E402

HARDWARE = HardwareProfile(name="synthetic", price_per_hour_usd=1.0,
                           effective_tflops=100.0, vram_gb=48.0,
                           measured=False, source="synthetic test fixture")


def _specs():
    """A small parent and an identical target. Geometry is not what is tested."""
    from aadistill.initialization.specs.arch import ArchSpec

    parent = ArchSpec.of("qwen3", {
        "hidden_size": 256, "num_hidden_layers": 4,
        "intermediate_size": 512, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 32, "vocab_size": 1024,
        "max_position_embeddings": 4096, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
    })
    return parent, parent


def _reducible():
    """A parent and a target that EVERY shipped operator can plan against.

    `_specs()` returns parent == target, which DEPTH rightly refuses -- there
    is nothing to remove. A registry walk needs a target that reduces every
    structural field, or it silently exercises only the operators that happen
    to tolerate a no-op.
    """
    parent, _ = _specs()
    target = parent.replace(num_hidden_layers=2, intermediate_size=256,
                            num_attention_heads=4, hidden_size=128)
    return parent, target


class _ForwardOnly(OperatorImplementation):
    """Declares forwards and no statistics. The case that priced at zero."""

    impl_id = "synthetic.forward_only_v0"
    kind = "ATTENTION"
    version = 0
    description = "synthetic: N forward passes, no statistics"
    calibration = CalibrationNeed.FORWARD_LOGITS
    modifies = frozenset()
    preserves = frozenset()

    def __init__(self, passes: int):
        self._passes = passes

    def plan(self, spec, target, adapter, config=None):
        return OperatorPlan(impl_id=self.impl_id, result_spec=spec,
                            forward_passes=self._passes, stats_passes=0,
                            notes="synthetic")

    def apply(self, ctx):                                    # pragma: no cover
        raise NotImplementedError("synthetic: never applied")


def _cost(passes: int, *, tokens: int = 10_000):
    parent, target = _specs()
    return operator_cost(_ForwardOnly(passes), parent, target, QWEN3_ADAPTER,
                         calibration_tokens=tokens, seq_len=1024,
                         hardware=HARDWARE)


def test_a_plan_declaring_forwards_is_not_free():
    """The defect, stated as the thing it allowed."""
    cost = _cost(60_099)
    assert cost.flops > 0
    assert cost.gpu_seconds > 0


def test_zero_declared_forwards_stays_zero():
    """A weight-only operator must not acquire a forward cost."""
    cost = _cost(0)
    assert cost.flops == 0
    assert cost.gpu_seconds == 0


def test_cost_is_linear_in_declared_forward_passes():
    """Doubling the declared passes doubles the price. No magic scaling."""
    one, two = _cost(1), _cost(2)
    assert two.flops == pytest.approx(2 * one.flops)


def test_cost_is_linear_in_calibration_tokens():
    small = _cost(100, tokens=1_000)
    large = _cost(100, tokens=10_000)
    assert large.flops == pytest.approx(10 * small.flops)


def test_the_note_states_what_was_priced():
    """A reader must be able to see WHY a number is what it is."""
    cost = _cost(896 * 67)
    assert "60032" in cost.notes.replace(",", "")
    assert "forward passes" in cost.notes


def test_a_forward_heavy_plan_outprices_a_weight_only_one_by_orders():
    """Sanity: 60k forwards is not within noise of zero."""
    heavy, none = _cost(60_099), _cost(0)
    assert heavy.gpu_seconds > 1.0
    assert none.gpu_seconds == 0.0


def test_core_pricing_names_no_operator_or_dimension_of_this_round():
    """The forward-pass branch must be generic, checked on EXECUTABLE code.

    Not on the file text: `L40S` legitimately appears in `cost.py` as the name
    and provenance of a MEASURED hardware profile, which is the generic
    mechanism -- callers pass a profile in. Banning the string would be banning
    the evidence for a number.

    So this parses the module and inspects the constants that actually
    participate in computation, with docstrings excluded. What must not appear
    there is this round's operator or its dimensions: a special case for
    `attention.causal_kl_v1` would make the NEXT forward-heavy operator free
    again, which is exactly how this defect arrived.
    """
    import ast

    tree = ast.parse((REPO / "src/aadistill/runtime/cost.py").read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    live = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and id(n) not in docstrings]
    strings = " ".join(v for v in live if isinstance(v, str))
    numbers = {v for v in live if isinstance(v, int) and not isinstance(v, bool)}

    for banned in ("causal_kl_v1", "phase_c3", "c3"):
        assert banned not in strings, (
            f"core pricing names {banned!r} in executable code")
    for banned in (896, 60099, 60032):
        assert banned not in numbers, (
            f"core pricing hardcodes the dimension {banned}")


def test_the_forward_branch_reads_the_plan_and_nothing_else():
    """It prices from `plan.forward_passes`, not from who the operator is."""
    import ast
    import inspect

    from aadistill.runtime import cost as cost_mod

    src = inspect.getsource(cost_mod.operator_cost)
    tree = ast.parse(src.lstrip())
    #: Find the `elif plan.forward_passes:` arm and confirm its body derives
    #: flops from the plan, the tokens and the parent spec -- no impl_id.
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if "forward_passes" not in test or "stats_passes" in test:
            continue
        found = True
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        assert "impl_id" not in body, (
            "the generic forward branch must not branch on an implementation id")
        assert "forward_passes" in body and "calibration_tokens" in body
    assert found, "no `elif plan.forward_passes:` branch found in operator_cost"


def test_the_existing_depth_operator_is_unaffected():
    """The new branch must not change any price that already existed.

    `depth.causal_kl_greedy_v1` declares `forward_passes` too, but is
    intercepted by the branch above; every other shipped operator declares
    zero. So no historical estimate moves.
    """
    from aadistill.initialization.operators.register import BUILTIN_OPERATORS

    parent, target = _reducible()
    assert BUILTIN_OPERATORS, "no shipped operators to check"
    for impl in BUILTIN_OPERATORS:
        if impl.impl_id == "depth.causal_kl_greedy_v1":
            continue
        plan = impl.plan(parent, target, QWEN3_ADAPTER, {"n_calibration_items": 1})
        assert plan.forward_passes == 0, (
            f"{impl.impl_id} declares forwards; its price now changes and that "
            "needs a deliberate review, not a silent one")


# --- the two declarations are ADDITIVE, not alternatives --------------------


class _Both(_ForwardOnly):
    """Declares forwards AND statistics. `OperatorPlan` permits it."""

    impl_id = "synthetic.forward_and_stats_v0"

    def __init__(self, passes: int, stats: int):
        super().__init__(passes)
        self._stats = stats

    def plan(self, spec, target, adapter, config=None):
        return OperatorPlan(impl_id=self.impl_id, result_spec=spec,
                            forward_passes=self._passes,
                            stats_passes=self._stats, notes="synthetic")


def _both_cost(passes: int, stats: int, *, include_stats: bool = True):
    parent, target = _specs()
    return operator_cost(_Both(passes, stats), parent, target, QWEN3_ADAPTER,
                         calibration_tokens=10_000, seq_len=1024,
                         hardware=HARDWARE, include_stats=include_stats)


def test_declaring_both_prices_both():
    """An `if/elif` documents an exclusion the TYPE does not enforce.

    `OperatorPlan` exposes the two counts independently and forbids nothing, so
    the first operator to declare both would have had half its work priced at
    zero.
    """
    both = _both_cost(1_000, 1)
    forwards_only = _cost(1_000)
    assert both.flops == pytest.approx(forwards_only.flops)
    assert both.stats_seconds_low > 0, "the statistics half was dropped"


def test_the_forward_half_is_unaffected_by_the_stats_declaration():
    assert _both_cost(500, 1).flops == pytest.approx(_both_cost(500, 3).flops)


def test_more_declared_stats_passes_cost_more():
    """Do not assume one corpus-equivalent sweep is all anyone can declare."""
    one, three = _both_cost(0, 1), _both_cost(0, 3)
    assert three.stats_seconds_low == pytest.approx(3 * one.stats_seconds_low)
    assert three.stats_seconds_high == pytest.approx(3 * one.stats_seconds_high)


def test_operator_plan_does_not_forbid_declaring_both():
    """The premise of the additive branch, asserted rather than assumed.

    If `OperatorPlan` ever DOES enforce mutual exclusion, this fails and the
    additive pricing can be simplified -- which is the point of pinning it.
    """
    parent, _ = _specs()
    plan = OperatorPlan(impl_id="x", result_spec=parent,
                        forward_passes=5, stats_passes=7)
    assert (plan.forward_passes, plan.stats_passes) == (5, 7)


def test_shipped_operators_still_declare_exactly_one_kind():
    """So this change moves no existing number, which is why it is safe now."""
    from aadistill.initialization.operators.register import BUILTIN_OPERATORS

    parent, target = _reducible()
    checked = 0
    for impl in BUILTIN_OPERATORS:
        plan = impl.plan(parent, target, QWEN3_ADAPTER, {"n_calibration_items": 1})
        checked += 1
        assert not (plan.forward_passes and plan.stats_passes), (
            f"{impl.impl_id} declares both; its price changes and that needs a "
            "deliberate review")
    assert checked == len(BUILTIN_OPERATORS), "an operator was skipped"

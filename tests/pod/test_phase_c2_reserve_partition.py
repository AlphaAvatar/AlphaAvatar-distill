"""Runtime enforcement matches the accounting: three clocks, three purposes.

The pricing record separates the expected beam path, `beam_composition_risk`,
`baseline_rebuild_reserve` and the artifact-recovery reserve. The launcher used
to hand the beam a deadline of `base + every reserve`, which let the beam run
into the 27.665 minutes held for a missing-B rebuild. That is not a pricing
error — the ceiling is unchanged — it is an enforcement error: money the
accounting had partitioned was unpartitioned at runtime.

Four things are asserted here, and the last is the one a unit test alone would
miss: that the rebuild's clock is FRESH. A `Deadline` starts counting when it is
constructed, so passing the beam's object to the fallback would give the rebuild
whatever the beam left — and after a full-envelope search that is nothing.

Zero cost, CPU only, no pod.
"""

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
#: REPO itself, because the helpers that build a faithful B identity live in
#: `tests/autoinit/test_phase_c2_baseline` and importing them by package path
#: needs the repository root on `sys.path`. Relying on pytest's rootdir
#: insertion is the fragility that already makes one unrelated test fail to
#: collect when its directory is selected alone.
for _extra in (".", "src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str((REPO / _extra).resolve()) not in sys.path:
        sys.path.insert(0, str((REPO / _extra).resolve()))

LAUNCHER = REPO / "scripts/pod/autoinit_phase_c2_launch.py"
DRIVER = REPO / "scripts/pod/autoinit_phase_c2_driver.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registered():
    from aadistill.initialization.operators import attention_activation
    from experiments.phase_c2.search_space import register_c2_operators

    register_c2_operators()
    try:
        yield
    finally:
        attention_activation.unregister()


@pytest.fixture(scope="module")
def wired(registered):
    launcher = load(LAUNCHER, "c2_launch_rp")
    driver = load(DRIVER, "c2_driver_rp")
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/c2", "--session-commit", "0" * 40,
         "--bundle", "aad_00000000.bundle"])
    spec = launcher.spec(args).validate()

    from aadistill.infrastructure.budget import StepTime, plan_session
    from experiments.phase_c2.session import c2_hard_ceiling_usd

    b = spec.budget
    plan = plan_session(
        price_per_hour=1.09, authorized_usd=c2_hard_ceiling_usd(REPO),
        arms=b.arms, steps_per_arm=b.steps_per_arm,
        step_time=StepTime(seconds=b.step_seconds, source=b.step_source),
        setup_minutes=b.setup_minutes,
        eval_minutes_per_arm=b.eval_minutes_per_arm,
        transfer_minutes=b.transfer_minutes, other_phases=b.other_phases,
        contingency_fraction=b.contingency_fraction,
        soft_stop_reserves=b.soft_stop_reserves,
        artifact_recovery_reserve_minutes=b.artifact_recovery_reserve_minutes)
    ctx = SimpleNamespace(args=args, image_digest="img@sha256:abc",
                          price=1.09, spent_usd=0.0)
    parsed = driver.build_parser().parse_args(
        spec.driver_command(ctx, plan).split()[2:])
    return launcher, driver, spec, plan, parsed


def named(plan, name: str) -> float:
    return next(r.minutes for r in plan.soft_stop_reserves if r.name == name)


def base_minutes(plan) -> float:
    return next(p.minutes for p in plan.breakdown
                if p.name == "beam_search_depth_early")


# --- the partition ---------------------------------------------------------

def test_the_beam_deadline_excludes_the_baseline_rebuild_reserve(wired):
    _launcher, _driver, _spec, plan, parsed = wired
    base = base_minutes(plan)
    risk = named(plan, "beam_composition_risk")
    rebuild = named(plan, "baseline_rebuild_reserve")

    assert parsed.search_deadline_minutes == pytest.approx(base + risk, abs=0.01)
    #: Strictly less than what it used to get. The regression this prevents is
    #: exactly the difference.
    assert parsed.search_deadline_minutes < base + risk + rebuild
    assert parsed.search_deadline_minutes + rebuild == pytest.approx(
        base + risk + rebuild, abs=0.01)


def test_the_beam_risk_reserve_is_still_included(wired):
    """The repair partitions; it does not shrink the beam's own envelope."""
    _launcher, _driver, _spec, plan, parsed = wired
    base = base_minutes(plan)
    risk = named(plan, "beam_composition_risk")
    assert risk > 0
    assert parsed.search_deadline_minutes > base
    assert parsed.search_deadline_minutes - base == pytest.approx(risk, abs=0.01)


def test_the_beam_envelope_is_the_structural_search_bound(wired):
    """Derived from the record, and equal to the bound the space model computes.

    Two independent routes to one number: the pricing record's
    `base + beam_composition_risk`, and `bound().max_minutes` from the search
    space itself. If they ever disagree the price describes a different search.
    """
    _launcher, _driver, _spec, plan, parsed = wired
    from experiments.phase_c2.search_space import bound, c2_search1_space

    structural = bound(c2_search1_space(), statistic="max").max_minutes
    assert parsed.search_deadline_minutes == pytest.approx(structural, abs=0.01)


def test_the_baseline_reserve_is_passed_as_its_own_flag(wired):
    _launcher, _driver, _spec, plan, parsed = wired
    assert parsed.baseline_rebuild_minutes == pytest.approx(
        named(plan, "baseline_rebuild_reserve"), abs=1e-6)
    assert parsed.baseline_rebuild_minutes == pytest.approx(27.665, abs=1e-3)


def test_reserve_lookup_is_by_name_and_refuses_an_absent_one(wired):
    """Summing the reserves is what unpartitioned them; naming them is the fix."""
    launcher, _driver, _spec, plan, _parsed = wired
    from aadistill.infrastructure.session import SessionSpecError

    with pytest.raises(SessionSpecError, match="no 'nope' reserve"):
        launcher.reserve_minutes(plan, "nope")


def test_the_artifact_recovery_reserve_is_unavailable_to_scientific_work(wired):
    """It sits between the soft stop and the hard terminate, and nothing else
    may reach it: `afford` refuses work that would cross the soft stop."""
    _launcher, _driver, _spec, plan, parsed = wired
    assert plan.artifact_recovery_reserve_minutes == 30.0
    assert plan.hard_terminate_minutes - plan.soft_stop_minutes == \
        pytest.approx(30.0, abs=1e-6)
    #: Neither runtime allowance includes it.
    assert parsed.search_deadline_minutes < plan.soft_stop_minutes
    assert parsed.baseline_rebuild_minutes < 30.0
    #: And the driver is told the soft stop, not the hard limit, as its ceiling
    #: for starting work.
    assert parsed.soft_stop_usd < parsed.authorized_usd


def test_the_partition_does_not_move_the_accepted_ceiling(wired):
    """A partition repair adds no work, so it adds no money."""
    _launcher, _driver, _spec, plan, _parsed = wired
    from experiments.phase_c2.session import c2_hard_ceiling_usd

    assert plan.hard_terminate_usd == pytest.approx(15.0446, abs=5e-5)
    assert c2_hard_ceiling_usd(REPO) == 15.0446


# --- the two affordability checks ------------------------------------------

def test_the_beam_is_afforded_against_its_full_envelope_not_the_expectation(
        wired):
    """`--search-minutes` is the envelope, not the 300.16-minute trajectory.

    Approving the beam against its expected path would approve work its own
    deadline permits and the soft stop may not fund.
    """
    _launcher, _driver, _spec, plan, parsed = wired
    assert parsed.search_minutes == pytest.approx(
        parsed.search_deadline_minutes, abs=0.01)
    assert parsed.search_minutes > base_minutes(plan)


def test_the_driver_affords_the_beam_before_starting_it():
    """Source-level, because the alternative is a GPU run.

    The check must precede the search call; an affordability check taken
    afterwards is a receipt.
    """
    body = DRIVER.read_text()
    check = body.index('self.afford(self.a.search_minutes')
    search = body.index("found = run_phase_a_search(")
    assert check < search, "the beam is started before it is afforded"
    assert "full envelope" in body[check:check + 400]


# --- the rebuild's own clock and its own check -----------------------------

@dataclass
class Captured:
    deadline: object = None
    afforded: list = None


def fallback_with_capture(monkeypatch, tmp_path, *, rebuild_minutes=27.665,
                          afford=None):
    """A real `BaselineFallback` whose executor records what it was handed."""
    from experiments.phase_c2 import baseline as B
    from tests.autoinit.test_phase_c2_baseline import (  # noqa: F401
        b_identity, real_shaped_step,
    )

    captured = Captured(afforded=[])

    def fake_materialize(spec, **kwargs):
        captured.deadline = kwargs.get("deadline")
        return [real_shaped_step()]

    monkeypatch.setattr(B, "materialize_fixed_path", fake_materialize)

    def record_afford(minutes, what):
        captured.afforded.append((minutes, what))
        if afford is not None:
            afford(minutes, what)

    return B.BaselineFallback(
        adapter=object(), workdir=tmp_path, rebuild_minutes=rebuild_minutes,
        afford=record_afford, repo_root=REPO, device="cuda",
        say=lambda *_: None), captured


def b_absent_result():
    from tests.autoinit.test_phase_c2_baseline import other_leaf

    @dataclass
    class R:
        leaves: list

    return R([other_leaf()])


def test_the_fallback_takes_minutes_not_a_deadline(registered):
    """A `Deadline` starts at construction; accepting one would leak the beam's.

    The field is a NUMBER by design, and `rebuild_deadline` is None until the
    rebuild begins.
    """
    from experiments.phase_c2 import baseline as B

    fields = {f.name for f in B.BaselineFallback.__dataclass_fields__.values()}
    assert "rebuild_minutes" in fields
    assert "deadline" not in fields, (
        "the fallback accepts a pre-started clock again; the beam's would be "
        "spent by the time the rebuild is reached")
    #: and the driver passes the number, not the search's Deadline object.
    body = DRIVER.read_text()
    assert "rebuild_minutes=self.a.baseline_rebuild_minutes" in body
    assert "deadline=deadline" not in body.split("fallback = B.BaselineFallback")[1]


def test_the_baseline_clock_starts_only_when_the_rebuild_does(
        registered, tmp_path, monkeypatch):
    """Fresh, not inherited: the whole allowance is available at the rebuild."""
    fallback, captured = fallback_with_capture(monkeypatch, tmp_path)
    assert fallback.rebuild_deadline is None, "a clock existed before the rebuild"

    #: Simulate a beam that has already burned wall-clock time.
    time.sleep(0.05)
    fallback(b_absent_result(), lambda: None)

    assert fallback.rebuild_deadline is not None
    handed = captured.deadline
    assert handed is fallback.rebuild_deadline
    #: Essentially the full reserve remains — a leaked beam deadline would show
    #: close to zero, or a negative remainder.
    assert handed.seconds == pytest.approx(27.665 * 60, abs=1e-6)
    assert handed.remaining() > 27.6 * 60
    assert not handed.expired()


def test_a_beam_that_used_its_whole_envelope_still_gets_the_full_reserve(
        registered, tmp_path, monkeypatch):
    """The regression, stated as behaviour.

    Under the old wiring the fallback received the beam's `Deadline`. This
    constructs one that is already exhausted, proves the fallback does not
    accept it, and proves the rebuild still runs on a full clock.
    """
    from aadistill.initialization.planning.search import Deadline
    from experiments.phase_c2 import baseline as B

    spent = Deadline(seconds=0.0)
    assert spent.expired()
    with pytest.raises(TypeError):
        B.BaselineFallback(adapter=object(), workdir=tmp_path,
                           rebuild_minutes=27.665, afford=lambda *_: None,
                           deadline=spent)

    fallback, captured = fallback_with_capture(monkeypatch, tmp_path)
    fallback(b_absent_result(), lambda: None)
    assert not captured.deadline.expired()


def test_the_rebuild_is_afforded_before_anything_is_materialized(
        registered, tmp_path, monkeypatch):
    """Independently of the beam's check, and BEFORE the executor runs."""
    from experiments.phase_c2 import baseline as B

    order: list = []

    def refusing(minutes, what):
        order.append(("afford", minutes, what))
        raise B.BaselineError(
            f"{what} needs {minutes:.1f} min and the soft stop is reached")

    fallback, captured = fallback_with_capture(monkeypatch, tmp_path,
                                               afford=refusing)
    with pytest.raises(B.BaselineError, match="soft stop"):
        fallback(b_absent_result(), lambda: None)

    assert [o[0] for o in order] == ["afford"]
    assert order[0][1] == pytest.approx(27.665, abs=1e-6)
    assert "baseline rebuild" in order[0][2]
    assert captured.deadline is None, (
        "the executor ran after the affordability check refused")
    assert fallback.rebuild_deadline is None


def test_a_searched_b_costs_no_affordability_check_and_no_clock(
        registered, tmp_path, monkeypatch):
    """The conditional reserve is conditional. Nothing is spent when B is found."""
    from tests.autoinit.test_phase_c2_baseline import b_leaf

    @dataclass
    class R:
        leaves: list

    fallback, captured = fallback_with_capture(monkeypatch, tmp_path)
    assert fallback(R([b_leaf()]), lambda: None) == []
    assert captured.afforded == []
    assert fallback.rebuild_deadline is None
    assert captured.deadline is None


def test_the_rebuild_records_the_allowance_it_actually_ran_on(
        registered, tmp_path, monkeypatch):
    fallback, _captured = fallback_with_capture(monkeypatch, tmp_path)
    fallback(b_absent_result(), lambda: None)
    allowance = fallback.outcome["allowance"]
    assert allowance["reserve"] == "baseline_rebuild_reserve"
    assert allowance["minutes"] == pytest.approx(27.665, abs=1e-6)
    assert allowance["clock"]["budget_minutes"] == pytest.approx(27.665,
                                                                 abs=1e-3)
    assert allowance["clock"]["expired"] is False

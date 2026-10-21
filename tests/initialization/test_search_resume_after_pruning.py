"""A search interrupted after a pruned level resumes without re-executing it.

`prune_weights=False` keeps a pruned state's bytes; `_restore` then accepts its
PRUNED journal record exactly as it accepts a MEASURED one (bytes re-identified
from disk), so the resumed level re-ranks the same evaluations and prunes the
same states. These tests interrupt a real tiny search INSIDE a later level --
after an earlier level has pruned -- resume it from the same workdir, and check
that the result is the uninterrupted run's and that no completed expansion ran
twice. With the historical `prune_weights=True` the released states cannot be
restored, and the resume recomputes them: that behaviour is unchanged.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.planning.ranking import PARETO_V1, BeamSchedule  # noqa: E402
from aadistill.initialization.planning.search import BeamSearch, SearchConfig  # noqa: E402
from aadistill.initialization.specs.arch import get_adapter  # noqa: E402

from support.toy import make_items  # noqa: E402


class Stop(RuntimeError):
    pass


def _search(workdir, teacher, target_spec, eval_suite, suite_items, profile, *,
            prune_weights, stop_after=None):
    evaluator = StateEvaluator(eval_suite, suite_items)
    evaluator.prime_reference(teacher)
    items = make_items()
    calls = {"n": 0}

    def measurer(model, sha):
        if stop_after is not None and calls["n"] >= stop_after:
            raise Stop("simulated pod loss")
        calls["n"] += 1
        return evaluator.evaluate(model, sha)

    config = SearchConfig(
        run_id="resume_pruned", target_spec=target_spec,
        schedule=BeamSchedule("test.beam", 1, "test", warmup_levels=0, width=1),
        seed=4242, workdir=workdir, profiles=(profile,), policy=PARETO_V1,
        suite=eval_suite, prune_weights=prune_weights,
        #: declared, as every Stage-0 session declares it: an undeclared
        #: protocol is adopted from the first measurement and costs a resumed
        #: run one re-measured state by design (`_bind_measurement_protocol`)
        measurement_protocol_id=evaluator.measurement_protocol_id)
    search = BeamSearch(
        adapter=get_adapter("qwen3"), config=config,
        root_teacher_id="tiny-teacher", root_teacher_sha256="a" * 64,
        root_loader=lambda: teacher, calibration_loader=lambda p: items,
        measurer=measurer)
    return search, calls


def _shape(result):
    return ([(lv.generated, lv.ranking.selected_ids if lv.ranking else None)
             for lv in result.levels],
            sorted(result.states),
            [leaf.state_id for leaf in result.complete_leaves],
            {sid: s.evaluation.values for sid, s in result.states.items()
             if s.evaluation is not None})


def _interrupt_and_resume(tmp_path, fixtures, *, prune_weights):
    full, full_calls = _search(tmp_path / "full", *fixtures, prune_weights=prune_weights)
    reference = full.run()
    level0 = len(reference.levels[0].generated)
    assert len(reference.levels) >= 2 and level0 >= 2
    # stop one measurement into level 1: level 0 has been ranked and pruned
    first, first_calls = _search(tmp_path / "split", *fixtures,
                                 prune_weights=prune_weights, stop_after=level0 + 1)
    with pytest.raises(Stop):
        first.run()
    second, second_calls = _search(tmp_path / "split", *fixtures,
                                   prune_weights=prune_weights)
    resumed = second.run()
    return reference, full_calls["n"], first_calls["n"], second_calls["n"], resumed


@pytest.fixture
def fixtures(teacher, target_spec, eval_suite, suite_items, profile):
    return (teacher, target_spec, eval_suite, suite_items, profile)


def test_kept_bytes_resume_exactly_and_recompute_nothing(tmp_path, fixtures):
    reference, n_full, n_first, n_second, resumed = _interrupt_and_resume(
        tmp_path, fixtures, prune_weights=False)
    assert _shape(resumed) == _shape(reference)
    # every expansion measured exactly once across the two sessions
    assert n_first + n_second == n_full
    level0 = set(reference.levels[0].generated)
    pruned = [sid for sid, s in reference.states.items()
              if s.validity.value == "pruned" and sid in level0]
    assert pruned, "the fixture must prune something before the interruption"
    assert set(pruned) <= set(resumed.resumed)


def test_released_bytes_are_recomputed_as_before(tmp_path, fixtures):
    reference, n_full, n_first, n_second, resumed = _interrupt_and_resume(
        tmp_path, fixtures, prune_weights=True)
    assert _shape(resumed) == _shape(reference)
    assert n_first + n_second > n_full

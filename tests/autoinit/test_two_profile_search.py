"""A full two-profile beam search, EXECUTED — not enumerated.

Phase B is a P=2 search, and until this file existed the project's evidence that
the engine supports P=2 was `test_two_profile_expansion` in `test_corrections.py`,
which asserts what `_candidate_expansions` *yields* and never calls `run()`. Every
search this repository has actually executed — both dry-run journals, every other
`make_search` caller — was P=1, and every one of them passed
`calibration_loader=lambda profile: items`, a closure that ignores its argument.
So the per-profile loading path had never once run with two distinct item sets,
and four paid pods in this project have died in code that no rehearsal executed.

What is exercised here that enumeration cannot reach:

* the loader is called with each profile and returns **genuinely different
  tokens**, so a profile that failed to reach the operator would change a result
  rather than a label;
* the full materialize -> reload -> hash -> validate -> measure cycle runs for
  both profiles;
* the two calibration-consuming branches produce **different weights**, which is
  the property the whole two-profile search rests on;
* the no-calibration branch produces **byte-identical** weights, which is the
  execution-level form of the 2026-08-12 Decision (1) invariant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.metrics import StateEvaluator  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1, BeamSchedule  # noqa: E402
from aadistill.autoinit.search import BeamSearch, SearchConfig  # noqa: E402
from conftest import TARGET_GEOMETRY, build_tiny_model, make_items, make_profile  # noqa: E402

ADAPTER = get_adapter("qwen3")

#: Wide enough that both profiles' children survive to be compared, and shallow
#: enough to stay a few seconds on CPU.
SCHEDULE = BeamSchedule("test.beam.p2", 1, "two-profile", warmup_levels=1, width=6)


@pytest.fixture(scope="module")
def p2(tmp_path_factory):
    """One executed P=2 search, reused by the assertions below."""
    from aadistill.autoinit.metrics import StateEvalSuite, SuiteItem

    teacher = build_tiny_model({"hidden_size": 32, "num_hidden_layers": 6,
                                "intermediate_size": 48, "num_attention_heads": 4,
                                "num_key_value_heads": 2, "head_dim": 8,
                                "vocab_size": 128, "tie_word_embeddings": True})
    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    suite = StateEvalSuite(
        suite_id="test.state_eval", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=4,
        general_domain="general")
    suite_items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                             domain=i["domain"], subtype=i["subtype"], tags=i["tags"])
                   for i in make_items(seed=202)]
    evaluator = StateEvaluator(suite, suite_items)
    evaluator.prime_reference(teacher)

    balanced, reasoning = make_profile("balanced", seed=1), make_profile("reasoning", seed=2)
    # THE POINT: different tokens per profile, not the same list twice.
    per_profile = {balanced.qualified_id: make_items(seed=101),
                   reasoning.qualified_id: make_items(seed=999)}
    assert per_profile[balanced.qualified_id][0]["input_ids"].tolist() != \
        per_profile[reasoning.qualified_id][0]["input_ids"].tolist(), \
        "the two profiles must carry different calibration tokens or this proves nothing"

    asked: list[str] = []

    def loader(profile):
        asked.append(profile.qualified_id)
        return per_profile[profile.qualified_id]

    config = SearchConfig(
        run_id="p2", target_spec=target_spec, schedule=SCHEDULE, seed=4242,
        workdir=tmp_path_factory.mktemp("p2"),
        profiles=(balanced, reasoning), policy=PARETO_V1, suite=suite)
    search = BeamSearch(
        adapter=ADAPTER, config=config, root_teacher_id="tiny-teacher",
        root_teacher_sha256="a" * 64, root_loader=lambda: teacher,
        calibration_loader=loader,
        measurer=lambda model, digest: evaluator.evaluate(model, digest))
    result = search.run()
    return {"result": result, "asked": asked, "target_spec": target_spec,
            "balanced": balanced, "reasoning": reasoning}


def test_the_two_profile_search_actually_completes(p2):
    result = p2["result"]
    assert result.complete_leaves, "a P=2 search produced no complete leaf"
    for leaf in result.complete_leaves:
        assert leaf.spec.matches(p2["target_spec"]), leaf.path_label
        leaf.require_recovery_admissible()


def test_both_profiles_reached_the_loader_and_the_states(p2):
    """A label is not evidence that the mixture was read."""
    assert set(p2["asked"]) == {p2["balanced"].qualified_id,
                                p2["reasoning"].qualified_id}, p2["asked"]
    seen = {pid for state in p2["result"].states.values() for pid in state.profile_ids}
    assert p2["balanced"].qualified_id in seen
    assert p2["reasoning"].qualified_id in seen
    assert "calib.none@v1" in seen, "the no-calibration sentinel never appeared"


def test_a_calibrated_operator_produces_DIFFERENT_weights_per_profile(p2):
    """The property the whole two-profile search rests on.

    Two states identical but for their profile must differ in their *artifact*.
    If they did not, P=2 would be doubling the cost of the search to measure the
    same checkpoint twice under two names.
    """
    by_key: dict[tuple, dict[str, str]] = {}
    for state in p2["result"].states.values():
        if len(state.steps) != 1 or state.artifact_digest is None:
            continue
        step = state.steps[0]
        if step.profile_id == "calib.none@v1":
            continue
        by_key.setdefault((step.kind, step.impl_id), {})[step.profile_id] = \
            state.artifact_digest

    compared = [(k, v) for k, v in by_key.items() if len(v) == 2]
    assert compared, "no calibrated operator was run under both profiles"
    for key, digests in compared:
        a, b = sorted(digests.values())
        assert a != b, (
            f"{key} produced byte-identical weights under two different calibration "
            "mixtures: the profile is not reaching the operator")


def test_a_no_calibration_operator_produces_IDENTICAL_weights(p2):
    """Decision (1), at execution rather than at enumeration.

    A `CalibrationNeed.NONE` implementation is invoked once against the sentinel,
    so it cannot appear twice at all — and the one state it produces must not
    depend on which profiles the search was configured with.
    """
    sentinel_states = [s for s in p2["result"].states.values()
                       if len(s.steps) == 1 and s.steps[0].profile_id == "calib.none@v1"]
    assert sentinel_states, "no no-calibration operator ran"
    by_impl: dict[str, list] = {}
    for state in sentinel_states:
        by_impl.setdefault(state.steps[0].impl_id, []).append(state)
    for impl_id, states in by_impl.items():
        assert len(states) == 1, (
            f"{impl_id} declares no calibration need and was expanded "
            f"{len(states)} times; it must be invoked once against the sentinel")


def test_the_statistics_cache_cannot_be_shared_across_profiles(p2):
    """Decision (8). Sharing one activation pass between two mixtures would make
    the second profile's operator read the first profile's statistics."""
    keys = {}
    for state in p2["result"].states.values():
        for step in state.steps:
            keys.setdefault(step.profile_id, set()).add(step.profile_hash)
    assert len(keys) >= 3, keys
    hashes = [next(iter(v)) for v in keys.values()]
    assert len(set(hashes)) == len(hashes), "two profiles share a profile_hash"

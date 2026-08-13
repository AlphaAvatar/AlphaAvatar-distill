"""The abstraction must not have changed anything the project already froze.

Two frozen records are checked here against live code:

* **E8a's contribution depth map.** The 260-evaluation greedy search's per-round
  tables were journalled. Replaying them through ``greedy_removal`` must
  reproduce the frozen removal order and kept set exactly — including the
  tie-break, which is the part a later reader would otherwise have to take on
  trust.
* **The incumbent Stage-1 recipe.** ``composite.stage1_sandwich_v0`` must be
  ``init_student``, not a reimplementation of it, so the same inputs give
  bitwise-identical weights through the operator abstraction and through the
  direct call.

The 4B teacher is not on this box, so the production hashes (86fbba78..., the
E8a map's own inputs) cannot be recomputed here. What *is* checked is everything
that does not need the teacher: the frozen selection replayed from frozen
measurements, the positional map at the production layer counts, the parameter
arithmetic against both recorded counts, and bitwise equality of the wrapper
against the wrapped algorithm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.operators.base import OperatorContext, get_implementation  # noqa: E402
from aadistill.init.contribution import greedy_removal  # noqa: E402
from aadistill.init.sandwich import depth_span_map, init_student  # noqa: E402

ADAPTER = get_adapter("qwen3")
E8A_DIR = REPO / "artifacts/stage1/e8_depth_search"

TEACHER_36 = ArchSpec.of("qwen3", dict(
    hidden_size=2560, num_hidden_layers=36, intermediate_size=9728,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))
TARGET_596M = ArchSpec.of("qwen3", dict(
    hidden_size=1024, num_hidden_layers=28, intermediate_size=3072,
    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))


@pytest.mark.skipif(
    not (REPO / "artifacts/stage1/qwen3_0p6b_init_v0/manifest.json").is_file(),
    reason="the init manifest is a gitignored artifact; a pod stages the "
           "checkpoint files, not the manifest")
def test_parameter_arithmetic_matches_both_frozen_counts():
    """The cost model prices states it never builds; the arithmetic must be exact."""
    assert ADAPTER.param_count(TEACHER_36) == 4_022_468_096
    assert ADAPTER.param_count(TARGET_596M) == 596_049_920
    manifest = json.loads((REPO / "artifacts/stage1/qwen3_0p6b_init_v0/manifest.json")
                          .read_text())
    assert manifest["teacher"]["num_parameters"] == ADAPTER.param_count(TEACHER_36)
    assert manifest["student"]["num_parameters"] == ADAPTER.param_count(TARGET_596M)


@pytest.mark.skipif(not (E8A_DIR / "e8_frozen_depth_map.json").is_file(),
                    reason="E8a frozen depth map not present")
def test_the_positional_map_still_removes_the_layers_the_record_says():
    frozen = json.loads((E8A_DIR / "e8_frozen_depth_map.json").read_text())
    kept = [s["representative"] for s in depth_span_map(36, 28)]
    removed = sorted(set(range(36)) - set(kept))
    assert removed == frozen["positional_removed"] == [5, 7, 9, 11, 13, 15, 17, 19]
    assert len(kept) == 28


@pytest.mark.skipif(not (E8A_DIR / "rounds.jsonl").is_file(),
                    reason="E8a search journal not present")
def test_e8a_depth_map_replays_from_its_frozen_rounds():
    """The frozen map is re-derived from the frozen measurements, not asserted.

    ``score_fn`` reads the recorded table instead of running the teacher, so this
    checks the *selection rule* — iterative greedy argmin with a lower-index
    tie-break — against 260 real measurements. A change to the rule breaks it
    even though every number is unchanged.
    """
    frozen = json.loads((E8A_DIR / "e8_frozen_depth_map.json").read_text())
    rounds = [json.loads(line) for line in (E8A_DIR / "rounds.jsonl").read_text()
              .splitlines() if line.strip()]
    assert len(rounds) == 8
    assert sum(r["n_candidates"] for r in rounds) == frozen["selector"][
        "subset_evaluations"] == 260

    table = {}
    for record in rounds:
        before = frozenset(record["removed_before"])
        for entry in record["table"]:
            table[before | {entry["candidate"]}] = entry["score"]

    replay = greedy_removal(lambda skip: table[frozenset(skip)], n_layers=36,
                            n_remove=8)
    assert replay["removal_order"] == frozen["removal_order"] == [
        2, 16, 3, 32, 20, 26, 15, 21]
    assert replay["removed"] == frozen["removed_teacher_layers"] == [
        2, 3, 15, 16, 20, 21, 26, 32]
    assert replay["kept"] == frozen["kept_teacher_layers"]
    assert replay["evaluations"] == 260
    # And the map really is a different decision from the positional one: one
    # shared removed layer out of eight.
    shared = set(replay["removed"]) & set(frozen["positional_removed"])
    assert shared == {15}


@pytest.mark.skipif(not (E8A_DIR / "rounds.jsonl").is_file(),
                    reason="E8a search journal not present")
def test_the_frozen_map_is_what_an_explicit_depth_map_would_build():
    """The map feeds Stage 1 through `kept_layers`, unchanged by the abstraction."""
    from aadistill.init.sandwich import explicit_depth_map

    frozen = json.loads((E8A_DIR / "e8_frozen_depth_map.json").read_text())
    spans = explicit_depth_map(frozen["kept_teacher_layers"], 36)
    assert len(spans) == 28
    assert [s["representative"] for s in spans] == frozen["kept_teacher_layers"]
    covered = [t for s in spans for t in range(*s["teacher_span"])]
    assert covered == list(range(36)), "every teacher layer belongs to exactly one span"


def test_the_composite_operator_is_the_incumbent_recipe_bitwise(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """Same inputs, same weights — through the abstraction and through the call.

    If this drifts, ``composite.stage1_sandwich_v0`` no longer denotes the
    algorithm that produced 86fbba78..., and every manifest citing that id
    becomes wrong about what it ran.
    """
    from aadistill.autoinit.operators._common import collect_activation_stats

    state = collect_activation_stats(
        ADAPTER, teacher, (i["input_ids"] for i in calibration_items), "cpu")

    direct = ADAPTER.build_model(
        ADAPTER.build_config(teacher.config, target_spec), torch.float32, 999)
    diag = init_student(teacher, direct, state)

    impl = get_implementation("composite.stage1_sandwich_v0")
    outcome = impl.execute(OperatorContext(
        adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
        target_spec=target_spec, profile=profile,
        calibration_items=calibration_items, seed=999,
        config={"activation_state": state, "verify_full_assignment": True}))

    for (name, a), (_, b) in zip(direct.named_parameters(),
                                 outcome.model.named_parameters()):
        assert torch.equal(a, b), f"{name} differs between the wrapper and init_student"
    assert outcome.trace["kept_layers"] == diag["kept_teacher_layers"]
    assert outcome.trace["source"] == diag["depth_map_source"] == \
        "positional_pairwise_merge"


def test_the_composite_accepts_a_contribution_map_as_the_single_variable(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """E8's design: only `kept_layers` changes between control and treatment."""
    from aadistill.autoinit.operators._common import collect_activation_stats

    state = collect_activation_stats(
        ADAPTER, teacher, (i["input_ids"] for i in calibration_items), "cpu")
    impl = get_implementation("composite.stage1_sandwich_v0")

    def run(kept):
        return impl.execute(OperatorContext(
            adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
            target_spec=target_spec, profile=profile,
            calibration_items=calibration_items, seed=5,
            config={"activation_state": state, "kept_layers": kept}))

    positional = [s["representative"] for s in depth_span_map(6, 4)]
    control = run(None)
    replayed = run(positional)
    # Feeding the positional map's own representatives back reproduces it bitwise,
    # which is what makes a depth-map experiment single-variable.
    for (name, a), (_, b) in zip(control.model.named_parameters(),
                                 replayed.model.named_parameters()):
        assert torch.equal(a, b), name

    treatment = run([0, 2, 3, 5])
    assert treatment.trace["kept_layers"] == [0, 2, 3, 5]
    assert treatment.trace["source"] == "explicit_kept_layers"


def test_the_causal_depth_wrapper_does_not_change_the_greedy_rule(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """The operator's kept set equals a direct `greedy_removal` on the same scores."""
    from aadistill.init.contribution import (
        bypassed_blocks, distortion, domain_balanced_score)

    impl = get_implementation("depth.causal_kl_greedy_v1")
    outcome = impl.execute(OperatorContext(
        adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
        target_spec=target_spec, profile=profile,
        calibration_items=calibration_items, seed=1))

    teacher.config.use_cache = False
    domains: dict[str, list[str]] = {}
    for item in calibration_items:
        subs = domains.setdefault(item["domain"], [])
        if item["subtype"] not in subs:
            subs.append(item["subtype"])
    with torch.no_grad():
        refs = [teacher(i["input_ids"]).logits[0, :-1].float() for i in calibration_items]

        def score(skip):
            per: dict[str, list[float]] = {}
            with bypassed_blocks(teacher, skip):
                for item, ref in zip(calibration_items, refs):
                    abl = teacher(item["input_ids"]).logits[0, :-1].float()
                    kl = distortion(ref, abl, item["input_ids"][0, 1:]).as_dict()["kl"]
                    per.setdefault(item["subtype"], []).append(kl)
            means = {k: sum(v) / len(v) for k, v in per.items()}
            return domain_balanced_score(means, {k: sorted(v) for k, v in domains.items()})[0]

        expected = greedy_removal(score, n_layers=6, n_remove=2)
    assert outcome.trace["kept_layers"] == expected["kept"]
    assert outcome.trace["removal_order"] == expected["removal_order"]
